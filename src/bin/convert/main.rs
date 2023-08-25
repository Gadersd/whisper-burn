use whisper::model::{*, load::*};

use burn::{
    module::Module, 
    tensor::{
        self, 
        backend::{self, Backend},
        Tensor,
        Int, 
    },
};

use burn_tch::{TchBackend, TchDevice};

use burn::config::Config;
use burn::record::{self, Recorder, DefaultRecorder};

fn save_whisper<B: Backend>(whisper: Whisper<B>, name: &str) -> Result<(), record::RecorderError> {
    DefaultRecorder::new()
    .record(
        whisper.into_record(),
        name.into(),
    )
}

use std::env;

fn main() {
    let model_name = match env::args().nth(1) {
        Some(name) => name,
        None => {
            eprintln!("Model dump folder not provided");
            return;
        }
    };

    type Backend = TchBackend<f32>;
    let device = TchDevice::Cpu;

    let (whisper, whisper_config): (Whisper<Backend>, WhisperConfig) = match load_whisper(&model_name) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Error loading model {}: {}", model_name, e);
            return;
        }
    };

    println!("Saving model...");
    if let Err(e) = save_whisper(whisper, &model_name) {
        eprintln!("Error saving model {}: {}", model_name, e);
        return;
    }

    println!("Saving config...");
    if let Err(e) = whisper_config.save(&format!("{}.cfg", model_name)) {
        eprintln!("Error saving config for {}: {}", model_name, e);
        return;
    }

    println!("Finished.");
}