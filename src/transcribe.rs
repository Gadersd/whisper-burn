use crate::model::*;
use crate::helper::*;
use crate::token::{self, *};
use crate::audio::{prep_audio, max_waveform_samples};

use num_traits::ToPrimitive;

use burn::{
    config::Config, 
    module::Module, 
    tensor::{
        self, 
        backend::{self, Backend},
        Data, 
        Tensor,
        Int, 
        Float, 
    },
};

pub fn waveform_to_text<B: Backend>(whisper: &Whisper<B>, bpe: &Gpt2Tokenizer, waveform: Vec<f32>, sample_rate: usize) -> token::Result<(String, Vec<usize>)> {
    let device = whisper.devices()[0].clone();

    let n_ctx_max_encoder = whisper.encoder_ctx_size();
    let padding = 10;
    let n_waveform_samples_per_window = max_waveform_samples(n_ctx_max_encoder - padding);

    let mel_iter = waveform_to_mel_tensor(waveform, sample_rate, n_waveform_samples_per_window, device);

    let mut text = String::new();
    let mut tokens: Vec<usize> = Vec::new();

    for (i, mel) in mel_iter.enumerate() {
        let mut prev_normal_tokens: Vec<_> = tokens.iter().rev().filter(|&&t| !bpe.is_special(t)).cloned().take(5).collect();
        prev_normal_tokens.reverse();
        //println!("Prev tokens: {:?} {}", prev_normal_tokens, bpe.decode(&prev_normal_tokens[..], false)?);

        let (new_text, new_tokens) = mels_to_text(whisper, bpe, mel, &prev_normal_tokens[..], padding)?;

        if let Some( (prev_index, curr_index) ) = find_chunk_overlap(&tokens[..], &new_tokens[..], 40, 3) {
            tokens.truncate(prev_index);
            tokens.extend(&new_tokens[curr_index..]);
        } else {
            tokens.extend(new_tokens);
        }

        text = bpe.decode(&tokens[..], true)?;
        println!("Chunk {}: {}\n", i, text);

        //text += &new_text;
    }

    Ok( (text, tokens) )
}

fn find_chunk_overlap(prev_tokens: &[usize], curr_tokens: &[usize], max_n_offsets: usize, min_n_overlaps: usize) -> Option<(usize, usize)> {
    let mut max_overlap = 0;
    let mut max_overlap_indices = (0, 0);
    let n_offsets = prev_tokens.len()
        .min(curr_tokens.len())
        .min(max_n_offsets);
    
    for offset in 0..n_offsets {
        let prev_start_index = prev_tokens.len() - 1 - offset;
        let mut overlap_iter = prev_tokens.iter()
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




fn waveform_to_mel_tensor<B: Backend>(waveform: Vec<f32>, sample_rate: usize, window_length_samples: usize, device: B::Device) -> impl Iterator<Item=Tensor<B, 3>> {
    let n_samples_per_tensor = window_length_samples;
    let chunk_overlap = sample_rate * 3;
    let shift = n_samples_per_tensor - chunk_overlap;
    let iter_len = (waveform.len() - n_samples_per_tensor) / shift + 1;

    (0..iter_len).into_iter().map(move |i| {
        let start = i * shift;
        let end = ( start + n_samples_per_tensor ).min(waveform.len());

        let slice = &waveform[start..end];

        let waveform = Tensor::from_floats(
            tensor::Data::new(slice.to_vec(), [slice.len()].into())
        ).to_device(&device);

        let mels = prep_audio(waveform.unsqueeze(), sample_rate as f64);

        mels
    })
}

fn mels_to_text<B: Backend>(whisper: &Whisper<B>, bpe: &Gpt2Tokenizer, mels: Tensor<B, 3>, prev_normal_tokens: &[usize], padding: usize) -> token::Result<(String, Vec<usize>)> {
    let device = mels.device();

    let n_ctx_max_encoder = whisper.encoder_ctx_size();
    let n_ctx_max_decoder = whisper.decoder_ctx_size();

    let [n_channel, n_mel, n_ctx] = mels.dims();
    if n_ctx + padding > n_ctx_max_encoder {
        println!("Audio has length of {} which exceeds maximum length {}. It will be clipped.", n_ctx + padding, n_ctx_max_encoder);
    }
    
    // the zero padding helps whisper determine end of text
    let mels = Tensor::cat(vec![mels.slice([0..1, 0..n_mel, 0..(n_ctx).min(n_ctx_max_encoder - padding)]), 
        Tensor::zeros_device([1, n_mel, padding ], &device)], 2);

    let start_token = bpe.special_token(SpecialToken::StartofTranscript).unwrap();
    let transcription_token = bpe.special_token(SpecialToken::Transcribe).unwrap();
    let start_of_prev_token = bpe.special_token(SpecialToken::StartofPrev).unwrap();
    let first_timestamp_token = bpe.special_token(SpecialToken::Timestamp(0.0)).unwrap();
    let end_token = bpe.special_token(SpecialToken::EndofText).unwrap();

    // including the prev tokens causes whisper to hallucinate, repeating itself and failing to determine end of text
    /*let mut tokens: Vec<usize> = iter::once(start_of_prev_token)
        .chain(prev_normal_tokens.into_iter().cloned())
        .chain(iter::once(start_token))
        .chain(iter::once(transcription_token))
        .chain(iter::once(bpe.special_token(SpecialToken::Timestamp(0.0)).unwrap()))
        .collect();*/
    let mut tokens = vec![start_token, transcription_token, first_timestamp_token];

    let encoder_output = whisper.forward_encoder(mels);

    loop {
        if tokens.len() >= n_ctx_max_decoder {
            tokens.push(end_token);
            break;
        }

        let token_tensor = Tensor::from_ints(
                Data::from_usize(Data::new(tokens.clone(), [tokens.len()].into()))
            ).unsqueeze::<2>()
             .to_device(&device);

        let out = whisper.forward_decoder(token_tensor, encoder_output.clone());

        let [n_batch, n_token, n_dict] = out.dims();
        let last_row: Tensor<B, 1> = out.slice([0..1, (n_token - 1)..n_token]).flatten(0, 2);

        let token_id = last_row.clone().argmax(0).into_scalar().to_usize().unwrap();
        let token_logit = last_row.clone().slice([token_id..(token_id + 1)]).into_scalar().to_f64().unwrap();
        let eot_logit = last_row.slice([end_token..(end_token + 1)]).into_scalar().to_f64().unwrap();

        tokens.push(token_id);
        //println!("{}", bpe.decode(&[token_id], false)?);

        // if end of text confidence is great enough then stop
        if (eot_logit - token_logit).exp() > 0.8 {
            if token_id != end_token {
                tokens.push(end_token);
            }
            break;
        }

        let repeat_window_size = 2;
        let min_n_repeats = 4; // three times to charm, four to scorn
        if let Some(index_of_first_repeat) = find_repeated_tokens_index(&tokens[..], repeat_window_size, min_n_repeats) {
            let end = index_of_first_repeat + repeat_window_size;

            tokens.truncate(end);
            tokens.push(end_token);
            break;
        }
    }

    let text = bpe.decode(&tokens[..], true)?;

    return Ok( (text, tokens) );
}

fn find_repeated_tokens_index(tokens: &[usize], window_size: usize, min_repeat_count: usize) -> Option<usize> {
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

    let mut repeats = sliding_windows
        .filter(|(_, window)| window == &last_window);

    let n_repeats = repeats.clone().count();
    if n_repeats >= min_repeat_count {
        let first_repeat_index = repeats.next().unwrap().0;
        return Some(first_repeat_index);
    } else {
        return None
    };
}