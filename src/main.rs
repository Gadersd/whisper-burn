use std::collections::HashMap;

use whisper::model::*;
use whisper::helper::*;
use whisper::token;

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


use hound;

fn load_wave_audio_tensor<B: Backend>(filename: &str) -> hound::Result<(Tensor<B, 2>, usize)> {
    let mut reader = hound::WavReader::open(filename)?;
    let spec = reader.spec();

    let duration = reader.duration() as usize;
    let sample_rate = spec.sample_rate as usize;
    let channels = spec.channels as usize;

    type T = i16;

    let floats = reader
        .into_samples::<T>()
        .map(|s| s.map(|s| s as f32 / T::MAX as f32))
        .collect::<hound::Result<Vec<_>>>()?;

    let waveform = Tensor::from_floats(
        tensor::Data::new(floats, [duration, channels].into())
    ).transpose();

    return Ok( (waveform, sample_rate) );
}


use num_traits::ToPrimitive;
use whisper::audio::prep_audio;
use whisper::token::{Gpt2Tokenizer, SpecialToken};

fn waveform_to_text<B: Backend>(whisper: &Whisper<B>, waveform: Tensor<B, 2>, sample_rate: usize, max_tokens: usize) -> token::Result<String> {
    let device = waveform.device();
    
    let bpe = Gpt2Tokenizer::new()?;

    let n_ctx_max = whisper.encoder_ctx_size();
    let mels = prep_audio(waveform, sample_rate as f64);
    let [n_channel, n_mel, n_ctx] = mels.dims();
    if n_ctx > n_ctx_max {
        println!("Audio exceeds maximum length. Audio will be clipped.");
    }
    let mels = mels.slice([0..1, 0..n_mel, 0..(n_ctx.min(n_ctx_max))]);

    let start_token = bpe.special_token(SpecialToken::StartofTranscript).unwrap();
    let end_token = bpe.special_token(SpecialToken::EndofText).unwrap();

    let mut tokens: Vec<usize> = vec![start_token];
    let mut text = String::new();

    let encoder_output = whisper.forward_encoder(mels);

    for i in 0..max_tokens {
        let token_tensor = Tensor::from_ints(
                Data::from_usize(Data::new(tokens.clone(), [tokens.len()].into()))
            ).unsqueeze::<2>()
             .to_device(&device);

        let out = whisper.forward_decoder(token_tensor, encoder_output.clone());

        let [n_batch, n_token, n_dict] = out.dims();
        let last_row: Tensor<B, 1> = out.slice([0..1, (n_token - 1)..n_token]).flatten(0, 2);

        let token_id = last_row.argmax(0).into_scalar().to_usize().unwrap();
        if token_id == end_token {
            break;
        }

        tokens.push(token_id);

        let token_text = bpe.decode(&[token_id]).unwrap();
        println!("{token_text}");

        text += &token_text;
    }

    return Ok(text);
}

use burn::record::{Recorder, DefaultRecorder, RecorderError};

fn load_whisper_model_file<B: Backend>(config: &WhisperConfig, filename: &str) -> Result<Whisper<B>, RecorderError> {
    DefaultRecorder::new()
    .load(filename.into())
    .map(|record| config.init().load_record(record))
}

#[cfg(feature = "f16")]
type Elem = burn::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

use std::{env, process};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <audio file> <model name>", args[0]);
        process::exit(1);
    }

    let wav_file = &args[1];
    let model_name = &args[2];

    let device = if cfg!(target_os = "macos") {
        burn_tch::TchDevice::Mps
    } else {
        burn_tch::TchDevice::Cuda(0)
    };

    type Backend = burn_tch::TchBackend<Elem>;

    let whisper_config = match WhisperConfig::load(&format!("{}.cfg", model_name)) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Failed to load whisper config: {}", e);
            process::exit(1);
        }
    };

    let whisper = match load_whisper_model_file(&whisper_config, model_name) {
        Ok(whisper_model) => whisper_model,
        Err(e) => {
            eprintln!("Failed to load whisper model file: {}", e);
            process::exit(1);
        }
    };

    let (waveform, sample_rate): (Tensor<Backend, 2>, usize) = match load_wave_audio_tensor(wav_file) {
        Ok(tuple) => tuple,
        Err(e) => {
            eprintln!("Failed to load wav audio: {}", e);
            process::exit(1);
        }
    };
    
    let whisper = whisper.to_device(&device);
    let waveform = waveform.to_device(&device);

    let text = match waveform_to_text(&whisper, waveform, sample_rate, 50) {
        Ok(transcription) => transcription,
        Err(e) => {
            eprintln!("Failed to transcribe waveform: {}", e);
            process::exit(1);
        }
    };

    println!("Transcribed text: {}", text);
}
