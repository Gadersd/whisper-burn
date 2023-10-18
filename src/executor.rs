use std::collections::HashMap;

use burn::{
    module::Module, 
    tensor::{
        self,
        backend::{self, Backend},
        Data, Float, Int, Tensor,
        ElementConversion, 
    },
};

use crate::model::{Whisper, WhisperCache};

#[derive(Clone)]
pub struct WhisperExecutor<'a, B: Backend> {
    whisper: &'a Whisper<B>, 
    encoder_output: Option<Tensor<B, 3>>, 
    decoder_caches: HashMap<Vec<usize>, WhisperCache<B>>, 
}

impl<'a, B: Backend> WhisperExecutor<'a, B> {
    pub fn from_whisper(whisper: &'a Whisper<B>) -> WhisperExecutor<'a, B> {
        Self {
            whisper: whisper, 
            encoder_output: None, 
            decoder_caches: HashMap::new(), 
        }
    }

    pub fn process_audio(&mut self, x: Tensor<B, 3>) {
        self.encoder_output = Some( self.whisper.forward_encoder(x) );
        self.decoder_caches.clear();
    }

    pub fn get_next_token_prediction_logits(&mut self, tokens: &[usize]) -> Option<Tensor<B, 1>> {
        let encoder_output = self.encoder_output.clone()?;
        let n_tokens = tokens.len();

        // grab and remove corresponding cache
        let mut cache = self.decoder_caches.remove(&tokens[0..n_tokens-1]).unwrap_or(self.whisper.cache_empty());

        let token_tensor = self.tokens_to_tensor(tokens);
        let logits = self.whisper.forward_decoder_cache(token_tensor, encoder_output, &mut cache);

        // insert updated cache
        self.decoder_caches.insert(tokens.to_vec(), cache);

        return Some( logits.slice([0..1, n_tokens-1..n_tokens]).flatten(0, 2) );
    }

    fn tokens_to_tensor(&self, tokens: &[usize]) -> Tensor<B, 2, Int> {
        let device = &self.whisper.devices()[0];

        Tensor::from_data(Data::new(
            tokens.iter().map(|&v| v as i32).collect::<Vec<_>>(),
            [tokens.len()].into(),
        ).convert())
        .to_device(device)
        .unsqueeze()
    }
}