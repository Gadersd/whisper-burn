use crate::audio::{max_waveform_samples, prep_audio};
use crate::helper::*;
use crate::model::*;
use crate::token::{self, *};
use crate::beam;

use num_traits::ToPrimitive;

use std::iter;

use burn::{
    config::Config,
    module::Module,
    tensor::{
        self,
        backend::{self, Backend},
        Data, Float, Int, Tensor,
        ElementConversion, 
        activation::log_softmax, 
    },
};

pub fn waveform_to_text<B: Backend>(
    whisper: &Whisper<B>,
    bpe: &Gpt2Tokenizer, 
    lang: Language, 
    waveform: Vec<f32>,
    sample_rate: usize,
) -> token::Result<(String, Vec<usize>)> {
    let device = whisper.devices()[0].clone();

    let n_ctx_max_encoder = whisper.encoder_ctx_size();
    let padding = 10;
    let n_waveform_samples_per_window = max_waveform_samples(n_ctx_max_encoder - padding);

    let mel_iter =
        waveform_to_mel_tensor(waveform, sample_rate, n_waveform_samples_per_window, device);

    let mut text = String::new();
    let mut tokens: Vec<usize> = Vec::new();

    for (i, mel) in mel_iter.enumerate() {
        let mut prev_normal_tokens: Vec<_> = tokens
            .iter()
            .rev()
            .filter(|&&t| !bpe.is_special(t))
            .cloned()
            .take(5)
            .collect();
        prev_normal_tokens.reverse();
        //println!("Prev tokens: {:?} {}", prev_normal_tokens, bpe.decode(&prev_normal_tokens[..], false)?);

        let (new_text, new_tokens) =
            mels_to_text(whisper, bpe, lang, mel, &prev_normal_tokens[..], padding)?;

        if let Some((prev_index, curr_index)) =
            find_chunk_overlap(&tokens[..], &new_tokens[..], 40, 3)
        {
            tokens.truncate(prev_index);
            tokens.extend(&new_tokens[curr_index..]);
        } else {
            tokens.extend(new_tokens);
        }

        //tokens.extend(new_tokens);

        text = bpe.decode(&tokens[..], true)?;
        println!("Chunk {}: {}\n", i, text);

        //text += &new_text;
    }

    Ok((text, tokens))
}

fn find_chunk_overlap(
    prev_tokens: &[usize],
    curr_tokens: &[usize],
    max_n_offsets: usize,
    min_n_overlaps: usize,
) -> Option<(usize, usize)> {
    let mut max_overlap = 0;
    let mut max_overlap_indices = (0, 0);
    let n_offsets = prev_tokens.len().min(curr_tokens.len()).min(max_n_offsets);

    for offset in 0..n_offsets {
        let prev_start_index = prev_tokens.len() - 1 - offset;
        let mut overlap_iter = prev_tokens
            .iter()
            .skip(prev_start_index)
            .zip(curr_tokens.iter())
            .enumerate()
            .filter(|(_, (&old, &new))| old == new);

        let n_overlap = overlap_iter.clone().count();
        if n_overlap > max_overlap {
            max_overlap = n_overlap;

            let curr_overlap_index = overlap_iter.next().unwrap().0;
            let prev_overlap_index = prev_start_index + curr_overlap_index;
            max_overlap_indices = (prev_overlap_index, curr_overlap_index)
        }
    }

    if max_overlap >= min_n_overlaps {
        Some(max_overlap_indices)
    } else {
        None
    }
}

fn waveform_to_mel_tensor<B: Backend>(
    waveform: Vec<f32>,
    sample_rate: usize,
    window_length_samples: usize,
    device: B::Device,
) -> impl Iterator<Item = Tensor<B, 3>> {
    let chunk_overlap = sample_rate * 3;
    let n_samples_per_tensor = window_length_samples;
    let shift = n_samples_per_tensor.saturating_sub(chunk_overlap).max(1);
    let iter_len = waveform.len().saturating_sub(n_samples_per_tensor) / shift + 1;

    println!("iter len = {}", iter_len);

    (0..iter_len).into_iter().map(move |i| {
        let start = i * shift;
        let end = (start + n_samples_per_tensor).min(waveform.len());

        println!("Start = {}, END = {}", start, end);
        println!("len = {}", waveform.len());

        let slice = &waveform[start..end];

        let waveform = Tensor::from_floats(tensor::Data::new(slice.to_vec(), [slice.len()].into()))
            .to_device(&device);

        let mels = prep_audio(waveform.unsqueeze(), sample_rate as f64);

        mels
    })
}

use std::f32;

#[derive(Clone)]
struct BeamSearchToken {
    token: usize, 
    log_prob: f64, 
}                             

fn mels_to_text<B: Backend>(
    whisper: &Whisper<B>,
    bpe: &Gpt2Tokenizer,
    lang: Language, 
    mels: Tensor<B, 3>,
    prev_nonspecial_tokens: &[usize],
    padding: usize,
) -> token::Result<(String, Vec<usize>)> {
    let device = mels.device();

    let n_ctx_max_encoder = whisper.encoder_ctx_size();
    let n_ctx_max_decoder = whisper.decoder_ctx_size();

    let [n_channel, n_mel, n_ctx] = mels.dims();
    if n_ctx + padding > n_ctx_max_encoder {
        println!(
            "Audio has length of {} which exceeds maximum length {}. It will be clipped.",
            n_ctx + padding,
            n_ctx_max_encoder
        );
    }

    // the zero padding helps whisper determine end of text
    let mels = Tensor::cat(
        vec![
            mels.slice([0..1, 0..n_mel, 0..(n_ctx).min(n_ctx_max_encoder - padding)]),
            Tensor::zeros_device([1, n_mel, padding], &device),
        ],
        2,
    );

    let start_token = bpe.special_token(SpecialToken::StartofTranscript).unwrap();
    let transcription_token = bpe.special_token(SpecialToken::Transcribe).unwrap();
    let start_of_prev_token = bpe.special_token(SpecialToken::StartofPrev).unwrap();
    let lang_token = bpe.special_token(SpecialToken::Language(lang)).unwrap();
    let first_timestamp_token = bpe.special_token(SpecialToken::Timestamp(0.0)).unwrap();
    let end_token = bpe.special_token(SpecialToken::EndofText).unwrap();
    let notimestamp = bpe.special_token(SpecialToken::NoTimeStamps).unwrap();

    // including the prev tokens causes whisper to hallucinate, repeating itself and failing to determine end of text
    /*let mut tokens: Vec<usize> = iter::once(start_of_prev_token)
    .chain(prev_normal_tokens.into_iter().cloned())
    .chain(iter::once(start_token))
    .chain(iter::once(transcription_token))
    .chain(iter::once(bpe.special_token(SpecialToken::Timestamp(0.0)).unwrap()))
    .collect();*/

    let mut initial_tokens = if prev_nonspecial_tokens.len() > 0 {
        iter::once(start_of_prev_token).chain(prev_nonspecial_tokens.iter().cloned()).collect()
    } else {
        Vec::new()
    };

    let mut initial_tokens = Vec::new();

    initial_tokens.extend([start_token, lang_token, transcription_token, notimestamp]);

    let initial_tokens = initial_tokens.into_iter().map(|tok| BeamSearchToken {
        token: tok, 
        log_prob: 0.0, 
    }).collect();

    /*let initial_tokens: Vec<_> = [start_token, lang_token, transcription_token, notimestamp].into_iter().map(|tok| BeamSearchToken {
        token: tok, 
        logit: 0.0, 
    }).collect();*/

    type BeamNode = beam::BeamNode<BeamSearchToken>;

    let initial_tokens = BeamNode {
        seq: initial_tokens, 
        log_prob: 0.0, 
    };

    let encoder_output = whisper.forward_encoder(mels);

    /*let neg_infty = -f32::INFINITY;
    let mut nonspecial_mask: Vec<f32> = (0..bpe.vocab_size()).into_iter().map(|tok| /*if bpe.is_special(tok) {neg_infty} else {0.0}*/ 0.0).collect();
    //nonspecial_mask[end_token] = neg_infty;
    let nonspecial_mask = Tensor::from_floats(Data::new(
        nonspecial_mask,
        [bpe.vocab_size()].into(),
    )).to_device(&device);*/

    let beam_size = 5;
    let max_depth = 100;

    let beamsearch_is_finished = |toks: &[BeamSearchToken]| {
        if let Some(btok) = toks.last() {
            btok.token == end_token
        } else {
            false
        }
    };

    let beamsearch_next = |beams: &[BeamNode]| {
        // convert tokens into tensor
        let max_seq_len = beams.iter().map(|beam| beam.seq.len()).max().unwrap_or(0);
        let flattened_tokens: Vec<_> = beams
            .iter()
            .flat_map(|beam| {
                let additional_tokens = max_seq_len - beam.seq.len();
                beam.seq.iter().map(|btok| btok.token).chain( iter::once(0).cycle().take(additional_tokens) )
            })
            .collect();

        let token_tensor = Tensor::from_ints(Data::from_usize(Data::new(
            flattened_tokens,
            [beams.len(), max_seq_len].into(),
        )))
        .to_device(&device);

        let logits = whisper.forward_decoder(token_tensor, encoder_output.clone().repeat(0, beams.len()));
        let log_probs = log_softmax(logits, 2);

        let [n_batch, n_token, n_dict] = log_probs.dims();
        let beam_log_probs = beams.iter().enumerate().map(|(i, beam)| {
            let batch = i;
            let token_index = beam.seq.len() - 1;

            log_probs.clone().slice([batch..batch + 1, token_index..token_index + 1]).flatten::<1>(0, 2).into_data().value
        });

        let continuations = beam_log_probs
            .zip(beams)
            .map(|(log_probs, beam)| {
                (log_probs
                    .into_iter()
                    .map(|log_prob| log_prob.elem::<f64>())
                    .enumerate()
                    .map(|(token_id, log_prob)| {
                        (
                            BeamSearchToken {
                                token: token_id, 
                                log_prob: log_prob, 
                            }, 
                            beam.log_prob + log_prob,  
                        )
                    }
                    )
                    .collect(), end_token)
            }).collect();

        continuations
    };

    let tokens: Vec<_> = beam::beam_search(vec![initial_tokens], beamsearch_next, beamsearch_is_finished, beam_size, max_depth)
        .into_iter()
        .map(|btok| btok.token)
        .collect();

    /*let mut tokens: Vec<_> = [start_token, lang_token, transcription_token, notimestamp].to_vec();

    loop {
        if tokens.len() >= n_ctx_max_decoder {
            tokens.push(end_token);
            break;
        }

        let token_tensor = Tensor::from_ints(Data::from_usize(Data::new(
            tokens.clone(),
            [tokens.len()].into(),
        )))
        .unsqueeze::<2>()
        .to_device(&device);

        let out = whisper.forward_decoder(token_tensor, encoder_output.clone());

        let [n_batch, n_token, n_dict] = out.dims();
        let last_row: Tensor<B, 1> = out.slice([0..1, (n_token - 1)..n_token]).flatten(0, 2);

        let token_id = last_row.clone().argmax(0).into_scalar().to_usize().unwrap();
        let token_logit = last_row
            .clone()
            .slice([token_id..(token_id + 1)])
            .into_scalar()
            .to_f64()
            .unwrap();
        let eot_logit = last_row
            .slice([end_token..(end_token + 1)])
            .into_scalar()
            .to_f64()
            .unwrap();

        tokens.push(token_id);
        //println!("{}", bpe.decode(&[token_id], false)?);

        // if end of text confidence is great enough then stop
        if (eot_logit - token_logit).exp() > 0.5 {
            if token_id != end_token {
                tokens.push(end_token);
            }
            break;
        }

        let repeat_window_size = 5;
        let min_n_repeats = 4; // three times to charm, four to scorn
        /*println!("{}", bpe.decode(&tokens[..], false).unwrap());
        if let Some(period) = repetition_period(&tokens[..], min_n_repeats) {
            println!("period = {}", period);
            let end = first_repetition_end(&tokens[..], period);

            tokens.truncate(end);
            tokens.push(end_token);
            break;
        }*/
        if let Some( (index_of_first_repeat, end) ) =
            find_repeated_tokens_index(&tokens[..], repeat_window_size, min_n_repeats)
        {
            //let end = index_of_first_repeat + repeat_window_size;

            tokens.truncate(end);
            tokens.push(end_token);
            break;
        }
    }*/

    let text = bpe.decode(&tokens[..], true)?;

    return Ok((text, tokens));
}

fn first_repetition_end(tokens: &[usize], period: usize) -> usize {
    for i in (period..tokens.len() - period).into_iter().rev() {
        if tokens[i - period..i] != tokens[i..i + period] {
            return i + 1;
        }
    }

    period
}

fn repetition_period(
    tokens: &[usize], 
    min_repetitions: usize, 
) -> Option<usize> {
    for i in (0..tokens.len()).into_iter().rev() {
        let period = tokens.len() - i;

        if i / period < min_repetitions {
            return None;
        }

        if (0..min_repetitions).into_iter().all(|j| {
            let e = i - period * j;
            let s = e - period;

            tokens[s..e] == tokens[i..i + period]
        }) {
            return Some(period);
        }
    }

    None
}

fn find_repeated_tokens_index(
    tokens: &[usize],
    window_size: usize,
    min_repeat_count: usize,
) -> Option<(usize, usize)> {
    // the last window isn't checked or overlapped with itself
    if 2 * window_size > tokens.len() {
        return None;
    }

    let last_index = tokens.len() - window_size;
    let last_window = &tokens[last_index..];

    let sliding_windows = (0..=(last_index - window_size))
        .into_iter()
        .map(|i| &tokens[i..(i + window_size)])
        .enumerate();

    let mut repeats = sliding_windows.filter(|(_, window)| window == &last_window);

    let n_repeats = repeats.clone().count();
    if n_repeats >= min_repeat_count {
        let first_repeat_index = repeats.next().unwrap().0;
        let end = repeats.next().unwrap().0;
        return Some( (first_repeat_index, end) );
    } else {
        return None;
    };
}
