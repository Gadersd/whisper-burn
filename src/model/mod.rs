pub mod load;

use crate::cache::TensorCache;

use std::f32::NEG_INFINITY;

use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        self,
        conv::{Conv1d, Conv1dConfig, Conv1dRecord},
        PaddingConfig1d,
        LayerNormConfig, 
        LayerNorm, 
    },
    tensor::{activation::softmax, backend::Backend, module::embedding, Distribution, Int, Tensor},
};

pub type WhisperCache<B> = TextDecoderCache<B>;

#[derive(Config, Debug)]
pub struct WhisperConfig {
    audio_encoder_config: AudioEncoderConfig,
    text_decoder_config: TextDecoderConfig,
}

impl WhisperConfig {
    pub fn init<B: Backend>(&self) -> Whisper<B> {
        let n_audio_state = self.audio_encoder_config.n_audio_state;
        let n_text_state = self.text_decoder_config.n_text_state;

        assert!(
            n_audio_state == n_text_state,
            "Audio encoder state size {} must be equal to text decoder state size {}.",
            n_audio_state,
            n_text_state
        );

        let encoder = self.audio_encoder_config.init();
        let decoder = self.text_decoder_config.init();

        Whisper { encoder, decoder }
    }
}

#[derive(Module, Debug)]
pub struct Whisper<B: Backend> {
    encoder: AudioEncoder<B>,
    decoder: TextDecoder<B>,
}

impl<B: Backend> Whisper<B> {
    pub fn forward(&self, mel: Tensor<B, 3>, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.decoder.forward(tokens, self.encoder.forward(mel))
    }

    pub fn forward_encoder(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.forward(mel)
    }

    pub fn forward_decoder(
        &self,
        tokens: Tensor<B, 2, Int>,
        encoder_output: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.decoder.forward(tokens, encoder_output)
    }

    pub fn forward_decoder_cache(
        &self,
        tokens: Tensor<B, 2, Int>,
        encoder_output: Tensor<B, 3>,
        cache: &mut TextDecoderCache<B>,  
    ) -> Tensor<B, 3> {
        self.decoder.forward_cache(tokens, encoder_output, cache)
    }

    pub fn cache_empty(&self) -> TextDecoderCache<B> {
        self.decoder.cache_empty()
    }

    pub fn encoder_ctx_size(&self) -> usize {
        self.encoder.ctx_size()
    }

    pub fn decoder_ctx_size(&self) -> usize {
        self.decoder.ctx_size()
    }
}

#[derive(Config, Debug)]
pub struct TextDecoderConfig {
    n_vocab: usize,
    n_text_ctx: usize,
    n_text_state: usize,
    n_text_head: usize,
    n_text_layer: usize,
}

impl TextDecoderConfig {
    pub fn init<B: Backend>(&self) -> TextDecoder<B> {
        let token_embedding = Tensor::random(
            [self.n_vocab, self.n_text_state],
            Distribution::Normal(0.0, 1.0),
        )
        .into();
        let positional_embedding = Tensor::random(
            [self.n_text_ctx, self.n_text_state],
            Distribution::Normal(0.0, 1.0),
        )
        .into();
        let blocks: Vec<_> = (0..self.n_text_layer)
            .into_iter()
            .map(|_| {
                ResidualDecoderAttentionBlockConfig::new(self.n_text_state, self.n_text_head).init()
            })
            .collect();
        let ln = LayerNormConfig::new(self.n_text_state).init();

        let mask = attn_decoder_mask(self.n_text_ctx).into();

        let n_vocab = self.n_vocab;
        let n_text_ctx = self.n_text_ctx;

        TextDecoder {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
            n_vocab,
            n_text_ctx,
        }
    }
}

#[derive(Module, Debug)]
pub struct TextDecoder<B: Backend> {
    token_embedding: Param<Tensor<B, 2>>,
    positional_embedding: Param<Tensor<B, 2>>,
    blocks: Vec<ResidualDecoderAttentionBlock<B>>,
    ln: LayerNorm<B>,
    mask: Param<Tensor<B, 2>>,
    n_vocab: usize,
    n_text_ctx: usize,
}

impl<B: Backend> TextDecoder<B> {
    fn forward(&self, x: Tensor<B, 2, Int>, xa: Tensor<B, 3>) -> Tensor<B, 3> {
        let [n_batch, seq_len] = x.dims();

        assert!(
            seq_len <= self.n_text_ctx,
            "Token sequence length {} must not exceed {}.",
            seq_len,
            self.n_text_ctx
        );

        let x = embedding(self.token_embedding.val(), x)
            + self
                .positional_embedding
                .val()
                .slice([0..seq_len])
                .unsqueeze::<3>();

        //let mask = attn_decoder_mask(seq_len);

        let x = self.blocks.iter().fold(x, |x, block| block.forward(x, xa.clone(), self.mask.val()));

        let x = self.ln.forward(x);
        return x.matmul(self.token_embedding.val().transpose().unsqueeze::<3>());
    }

    fn forward_cache(&self, x: Tensor<B, 2, Int>, xa: Tensor<B, 3>, cache: &mut TextDecoderCache<B>) -> Tensor<B, 3> {
        let [n_batch, seq_len] = x.dims();

        assert!(
            seq_len <= self.n_text_ctx,
            "Token sequence length {} must not exceed {}.",
            seq_len,
            self.n_text_ctx
        );

        let x = embedding(self.token_embedding.val(), x)
            + self
                .positional_embedding
                .val()
                .slice([0..seq_len])
                .unsqueeze::<3>();

        //let mask = attn_decoder_mask(seq_len);

        let x = self.blocks.iter().zip(&mut cache.blocks).fold(x, |x, (block, cache)| block.forward_cache(x, xa.clone(), self.mask.val(), cache));
        //let x = self.blocks.iter().fold(x, |x, block| block.forward(x, xa.clone(), self.mask.val()));

        cache.out.forward_autoregressive(x, 1, |t| {
            let t = self.ln.forward(t);
            t.matmul(self.token_embedding.val().transpose().unsqueeze::<3>())
        })
    }

    pub fn cache_empty(&self) -> TextDecoderCache<B> {
        TextDecoderCache::empty(self.blocks.len())
    }

    fn ctx_size(&self) -> usize {
        self.n_text_ctx
    }
}

#[derive(Clone)]
pub struct TextDecoderCache<B: Backend> {
    blocks: Vec<ResidualDecoderAttentionBlockCache<B>>,
    out: TensorCache<B, 3>,
    layer2_out: Option<Tensor<B, 1>>, 
}

impl<B: Backend> TextDecoderCache<B> {
    fn empty(n_blocks: usize) -> Self {

        Self {
            blocks: (0..n_blocks).into_iter().map(|_| ResidualDecoderAttentionBlockCache::empty()).collect(), 
            out: TensorCache::empty(), 
            layer2_out: None, 
        }
    }
}

#[derive(Config, Debug)]
pub struct AudioEncoderConfig {
    n_mels: usize,
    n_audio_ctx: usize,
    n_audio_state: usize,
    n_audio_head: usize,
    n_audio_layer: usize,
}

impl AudioEncoderConfig {
    pub fn init<B: Backend>(&self) -> AudioEncoder<B> {
        let conv1 = Conv1dConfig::new(self.n_mels, self.n_audio_state, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .init();
        let gelu1 = nn::GELU::new();
        let conv2 = Conv1dConfig::new(self.n_audio_state, self.n_audio_state, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_stride(2)
            .init();
        let gelu2 = nn::GELU::new();
        let blocks: Vec<_> = (0..self.n_audio_layer)
            .into_iter()
            .map(|_| {
                ResidualEncoderAttentionBlockConfig::new(self.n_audio_state, self.n_audio_head)
                    .init()
            })
            .collect();
        let ln_post = LayerNormConfig::new(self.n_audio_state).init();
        let positional_embedding = Tensor::random(
            [self.n_audio_ctx, self.n_audio_state],
            Distribution::Normal(0.0, 1.0),
        )
        .into();
        let n_mels = self.n_mels;
        let n_audio_ctx = self.n_audio_ctx;

        AudioEncoder {
            conv1,
            gelu1,
            conv2,
            gelu2,
            blocks,
            ln_post,
            positional_embedding,
            n_mels,
            n_audio_ctx,
        }
    }
}

#[derive(Module, Debug)]
pub struct AudioEncoder<B: Backend> {
    conv1: Conv1d<B>,
    gelu1: nn::GELU,
    conv2: Conv1d<B>,
    gelu2: nn::GELU,
    blocks: Vec<ResidualEncoderAttentionBlock<B>>,
    ln_post: LayerNorm<B>,
    positional_embedding: Param<Tensor<B, 2>>,
    n_mels: usize,
    n_audio_ctx: usize,
}

impl<B: Backend> AudioEncoder<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, n_mels, n_ctx] = x.dims();

        assert!(
            n_mels == self.n_mels,
            "Audio mel spectrum size must be {}.",
            self.n_mels
        );
        assert!(
            n_ctx <= self.n_audio_ctx,
            "Audio length {} cannot exceed {}.",
            n_ctx,
            self.n_audio_ctx
        );

        let x = self.gelu1.forward(self.conv1.forward(x));
        let x = self.gelu2.forward(self.conv2.forward(x));

        let x = x.swap_dims(1, 2);
        let k = x.dims()[1];
        let x = x + self
            .positional_embedding
            .val()
            .slice([0..k])
            .unsqueeze::<3>();

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x);
        }

        return self.ln_post.forward(x);
    }

    fn ctx_size(&self) -> usize {
        self.n_audio_ctx
    }
}

#[derive(Config)]
pub struct ResidualEncoderAttentionBlockConfig {
    n_state: usize,
    n_head: usize,
}

impl ResidualEncoderAttentionBlockConfig {
    pub fn init<B: Backend>(&self) -> ResidualEncoderAttentionBlock<B> {
        let attn = MultiHeadSelfAttentionConfig::new(self.n_state, self.n_head).init();
        let attn_ln = LayerNormConfig::new(self.n_state).init();

        let mlp = MLPConfig::new(self.n_state).init();
        let mlp_ln = LayerNormConfig::new(self.n_state).init();

        ResidualEncoderAttentionBlock {
            attn,
            attn_ln,
            mlp,
            mlp_ln,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualEncoderAttentionBlock<B: Backend> {
    attn: MultiHeadSelfAttention<B>,
    attn_ln: LayerNorm<B>,
    mlp: MLP<B>,
    mlp_ln: LayerNorm<B>,
}

impl<B: Backend> ResidualEncoderAttentionBlock<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.attn_ln.forward(x), None);
        let x = x.clone() + self.mlp.forward(self.mlp_ln.forward(x));
        return x;
    }
}

#[derive(Config)]
pub struct ResidualDecoderAttentionBlockConfig {
    n_state: usize,
    n_head: usize,
}

impl ResidualDecoderAttentionBlockConfig {
    pub fn init<B: Backend>(&self) -> ResidualDecoderAttentionBlock<B> {
        let attn = MultiHeadSelfAttentionConfig::new(self.n_state, self.n_head).init();
        let attn_ln = LayerNormConfig::new(self.n_state).init();

        let cross_attn = MultiHeadCrossAttentionConfig::new(self.n_state, self.n_head).init();
        let cross_attn_ln = LayerNormConfig::new(self.n_state).init();

        let mlp = MLPConfig::new(self.n_state).init();
        let mlp_ln = LayerNormConfig::new(self.n_state).init();

        ResidualDecoderAttentionBlock {
            attn,
            attn_ln,
            cross_attn,
            cross_attn_ln,
            mlp,
            mlp_ln,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualDecoderAttentionBlock<B: Backend> {
    attn: MultiHeadSelfAttention<B>,
    attn_ln: LayerNorm<B>,
    cross_attn: MultiHeadCrossAttention<B>,
    cross_attn_ln: LayerNorm<B>,
    mlp: MLP<B>,
    mlp_ln: LayerNorm<B>,
}

impl<B: Backend> ResidualDecoderAttentionBlock<B> {
    fn forward(&self, x: Tensor<B, 3>, xa: Tensor<B, 3>, mask: Tensor<B, 2>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.attn_ln.forward(x), Some(mask));
        let x = x.clone() + self.cross_attn.forward(self.cross_attn_ln.forward(x), xa);
        let x = x.clone() + self.mlp.forward(self.mlp_ln.forward(x));
        return x;
    }

    fn forward_cache(&self, x: Tensor<B, 3>, xa: Tensor<B, 3>, mask: Tensor<B, 2>, cache: &mut ResidualDecoderAttentionBlockCache<B>) -> Tensor<B, 3> {
        let ln = self.attn_ln.forward(x.clone());
        let x = x + self.attn.forward(ln, Some(mask)/*, &mut cache.attn*/);

        //let x = x.clone() + self.attn.forward(self.attn_ln.forward(x), Some(mask));
        
        let x = x.clone() + self.cross_attn.forward_cache(self.cross_attn_ln.forward(x), xa, &mut cache.cross_attn);

        let x = cache.out.forward_autoregressive(x, 1, |t| {
            t.clone() + self.mlp.forward(self.mlp_ln.forward(t))
        });
        //let x = x.clone() + self.mlp.forward(self.mlp_ln.forward(x));

        return x;
    }
}

#[derive(Clone)]
pub struct ResidualDecoderAttentionBlockCache<B: Backend> {
    attn: MultiHeadSelfAttentionCache<B>,
    cross_attn: MultiHeadCrossAttentionCache<B>,
    out: TensorCache<B, 3>,
}

impl<B: Backend> ResidualDecoderAttentionBlockCache<B> {
    fn empty() -> Self {
        Self {
            attn: MultiHeadSelfAttentionCache::empty(), 
            cross_attn: MultiHeadCrossAttentionCache::empty(), 
            out: TensorCache::empty(), 
        }
    }
}

#[derive(Config)]
pub struct MLPConfig {
    n_state: usize,
}

impl MLPConfig {
    pub fn init<B: Backend>(&self) -> MLP<B> {
        let lin1 = nn::LinearConfig::new(self.n_state, 4 * self.n_state).init();
        let gelu = nn::GELU::new();
        let lin2 = nn::LinearConfig::new(4 * self.n_state, self.n_state).init();

        MLP { lin1, gelu, lin2 }
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    lin1: nn::Linear<B>,
    gelu: nn::GELU,
    lin2: nn::Linear<B>,
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.lin1.forward(x);
        let x = self.gelu.forward(x);
        let x = self.lin2.forward(x);

        return x;
    }

    pub fn forward_cache(&self, x: Tensor<B, 3>, cache: &mut MLPCache<B>) -> Tensor<B, 3> {
        cache.lin2.forward_autoregressive(x, 1, |t| self.forward(t))
    }
}

#[derive(Clone)]
pub struct MLPCache<B: Backend> {
    lin2: TensorCache<B, 3>,
}

impl<B: Backend> MLPCache<B> {
    fn empty() -> Self {
        Self {
            lin2: TensorCache::empty(), 
        }
    }
}

#[derive(Config)]
pub struct MultiHeadSelfAttentionConfig {
    n_state: usize,
    n_head: usize,
}

impl MultiHeadSelfAttentionConfig {
    fn init<B: Backend>(&self) -> MultiHeadSelfAttention<B> {
        assert!(
            self.n_state % self.n_head == 0,
            "State size {} must be a multiple of head size {}",
            self.n_state,
            self.n_head
        );

        let n_head = self.n_head;
        let query = nn::LinearConfig::new(self.n_state, self.n_state).init();
        let key = nn::LinearConfig::new(self.n_state, self.n_state)
            .with_bias(false)
            .init();
        let value = nn::LinearConfig::new(self.n_state, self.n_state).init();
        let out = nn::LinearConfig::new(self.n_state, self.n_state).init();

        MultiHeadSelfAttention {
            n_head,
            query,
            key,
            value,
            out,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadSelfAttention<B: Backend> {
    n_head: usize,
    query: nn::Linear<B>,
    key: nn::Linear<B>,
    value: nn::Linear<B>,
    out: nn::Linear<B>,
}

impl<B: Backend> MultiHeadSelfAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        let q = self.query.forward(x.clone());
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x);

        let wv = qkv_attention(q, k, v, mask, self.n_head);

        return self.out.forward(wv);
    }

    pub fn forward_cache(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2>>, cache: &mut MultiHeadSelfAttentionCache<B>) -> Tensor<B, 3> {
        let q = cache.query.forward_autoregressive(x.clone(), 1, |t| self.query.forward(t));
        let k = cache.key.forward_autoregressive(x.clone(), 1, |t| self.key.forward(t));
        let v = cache.value.forward_autoregressive(x, 1, |t| self.value.forward(t));

        let wv = qkv_attention(q, k, v, mask, self.n_head);

        return cache.out.forward_autoregressive(wv, 1, |t| self.out.forward(t));
    }
}

#[derive(Clone)]
pub struct MultiHeadSelfAttentionCache<B: Backend> {
    query: TensorCache<B, 3>,
    key: TensorCache<B, 3>,
    value: TensorCache<B, 3>,
    out: TensorCache<B, 3>,
}

impl<B: Backend> MultiHeadSelfAttentionCache<B> {
    fn empty() -> Self {
        Self {
            query: TensorCache::empty(),
            key: TensorCache::empty(),
            value: TensorCache::empty(),
            out: TensorCache::empty(),
        }
    }
}

#[derive(Config)]
pub struct MultiHeadCrossAttentionConfig {
    n_state: usize,
    n_head: usize,
}

impl MultiHeadCrossAttentionConfig {
    fn init<B: Backend>(&self) -> MultiHeadCrossAttention<B> {
        assert!(
            self.n_state % self.n_head == 0,
            "State size {} must be a multiple of head size {}",
            self.n_state,
            self.n_head
        );

        let n_head = self.n_head;
        let query = nn::LinearConfig::new(self.n_state, self.n_state).init();
        let key = nn::LinearConfig::new(self.n_state, self.n_state)
            .with_bias(false)
            .init();
        let value = nn::LinearConfig::new(self.n_state, self.n_state).init();
        let out = nn::LinearConfig::new(self.n_state, self.n_state).init();

        MultiHeadCrossAttention {
            n_head,
            query,
            key,
            value,
            out,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadCrossAttention<B: Backend> {
    n_head: usize,
    query: nn::Linear<B>,
    key: nn::Linear<B>,
    value: nn::Linear<B>,
    out: nn::Linear<B>,
}

use burn::tensor::ElementConversion;

impl<B: Backend> MultiHeadCrossAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, xa: Tensor<B, 3>) -> Tensor<B, 3> {
        let q = self.query.forward(x);
        let k = self.key.forward(xa.clone());
        let v = self.value.forward(xa);

        let wv = qkv_attention(q, k, v, None, self.n_head);

        return self.out.forward(wv);
    }

    pub fn forward_cache(&self, x: Tensor<B, 3>, xa: Tensor<B, 3>, cache: &mut MultiHeadCrossAttentionCache<B>) -> Tensor<B, 3> {
        let [_, n, _] = x.dims();
        //let q2 = self.query.forward(x.clone());
        let q = cache.query.forward_autoregressive(x.clone(), 1, |t| self.query.forward(t));
        let k = cache.key.forward_full(xa.clone(), |t| self.key.forward(t));
        let v = cache.value.forward_full(xa.clone(), |t| self.value.forward(t));

        let q2 = self.query.forward(x);
        let k2 = self.key.forward(xa.clone());
        let v2 = self.value.forward(xa);

        
        let diff = (q.clone() - q2.clone()).flatten::<1>(0, 2).abs().max().into_scalar().elem::<f64>();
        println!("Diff = {}", diff);

        let wv = qkv_attention(q2, k2, v2, None, self.n_head);

        return cache.out.forward_autoregressive(wv, 1, |t| self.out.forward(t));
    }
}

#[derive(Clone)]
pub struct MultiHeadCrossAttentionCache<B: Backend> {
    query: TensorCache<B, 3>,
    key: TensorCache<B, 3>,
    value: TensorCache<B, 3>,
    out: TensorCache<B, 3>,
}

impl<B: Backend> MultiHeadCrossAttentionCache<B> {
    fn empty() -> Self {
        Self {
            query: TensorCache::empty(), 
            key: TensorCache::empty(), 
            value: TensorCache::empty(), 
            out: TensorCache::empty(), 
        }
    }
}

pub fn qkv_attention<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    mask: Option<Tensor<B, 2>>,
    n_head: usize,
) -> Tensor<B, 3> {
    let [n_batch, n_qctx, n_state] = q.dims();
    let [_, n_ctx, _] = k.dims();

    let scale = (n_state as f64 / n_head as f64).powf(-0.25);
    let n_hstate = n_state / n_head;

    let q = q
        .reshape([n_batch, n_qctx, n_head, n_hstate])
        .swap_dims(1, 2)
        * scale;
    let k = k
        .reshape([n_batch, n_ctx, n_head, n_hstate])
        .swap_dims(1, 2)
        .transpose()
        * scale;
    let v = v
        .reshape([n_batch, n_ctx, n_head, n_hstate])
        .swap_dims(1, 2);

    let qk = q.matmul(k);

    // apply mask
    let qk = if let Some(mask) = mask {
        qk + mask.slice([0..n_qctx, 0..n_ctx]).unsqueeze::<4>()
    } else {
        qk
    };

    // normalize value weightings
    let w = softmax(qk, 3);
    let o = w.matmul(v).swap_dims(1, 2).flatten(2, 3);

    return o;
}

pub fn attn_decoder_mask<B: Backend>(seq_length: usize) -> Tensor<B, 2> {
    let mut mask = Tensor::<B, 2>::zeros([seq_length, seq_length]);

    for i in 0..(seq_length - 1) {
        let values = Tensor::<B, 2>::zeros([1, seq_length - (i + 1)]).add_scalar(NEG_INFINITY);
        mask = mask.slice_assign([i..i + 1, i + 1..seq_length], values);
    }

    return mask;
}
