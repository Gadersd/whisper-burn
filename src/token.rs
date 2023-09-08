use serde::ser::StdError;
use std::result;

use tokenizers::{AddedToken, Tokenizer};

pub type Result<T> = result::Result<T, Box<(dyn StdError + Send + Sync + 'static)>>;

pub struct Gpt2Tokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl Gpt2Tokenizer {
    pub fn new() -> Result<Self> {
        let mut tokenizer = tokenizers::Tokenizer::from_pretrained("gpt2", None)?;
        tokenizer.add_special_tokens(&construct_special_tokens());

        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let tokens = self.tokenizer.encode(text, true).unwrap();
        tokens.get_ids().iter().map(|t| *t as usize).collect()
    }

    pub fn special_token(&self, token: SpecialToken) -> Option<usize> {
        self.tokenizer
            .token_to_id(&token.to_string())
            .map(|t| t as usize)
    }

    pub fn decode(&self, tokens: &[usize], skip_special: bool) -> Result<String> {
        self.tokenizer
            .decode(tokens.iter().map(|t| *t as u32).collect(), skip_special)
    }

    pub fn is_special(&self, token: usize) -> bool {
        self.tokenizer
            .decode(vec![token as u32], true)
            .ok()
            .map(|s| s.is_empty())
            .unwrap_or(false)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}

const LANGUAGES: [&str; 98] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "ln", "ha", "ba",
    "jw", "su",
];

pub enum SpecialToken {
    EndofText,
    StartofTranscript,
    Translate,
    Transcribe,
    StartofLM,
    StartofPrev,
    NoSpeech,
    NoTimeStamps,
    Language(String),
    Timestamp(f64),
}

impl ToString for SpecialToken {
    fn to_string(&self) -> String {
        match self {
            SpecialToken::EndofText => "<|endoftext|>".into(),
            SpecialToken::StartofTranscript => "<|startoftranscript|>".into(),
            SpecialToken::Translate => "<|translate|>".into(),
            SpecialToken::Transcribe => "<|transcribe|>".into(),
            SpecialToken::StartofLM => "<|startoflm|>".into(),
            SpecialToken::StartofPrev => "<|startofprev|>".into(),
            SpecialToken::NoSpeech => "<|nospeech|>".into(),
            SpecialToken::NoTimeStamps => "<|notimestamps|>".into(),
            SpecialToken::Language(lang) => format!("<|{}|>", lang),
            SpecialToken::Timestamp(val) => format!("<|{:.2}|>", val),
        }
    }
}

fn construct_special_tokens() -> Vec<AddedToken> {
    const SPEC1: [&str; 2] = ["<|endoftext|>", "<|startoftranscript|>"];

    let lang_keys = LANGUAGES.iter().map(|lang| format!("<|{}|>", lang));

    const SPEC2: [&str; 6] = [
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
    ];

    let range_keys = (0..1501)
        .into_iter()
        .map(|i| i as f64 * 0.02)
        .map(|f| format!("<|{:.2}|>", f));

    SPEC1
        .into_iter()
        .map(String::from)
        .chain(lang_keys.into_iter())
        .chain(SPEC2.into_iter().map(String::from))
        .chain(range_keys.into_iter())
        .map(|tok| AddedToken::from(tok, true))
        .collect()
}
