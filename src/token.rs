use serde::ser::StdError;
use std::result;

use tokenizers::{AddedToken};

pub type Result<T> = result::Result<T, Box<(dyn StdError + Send + Sync + 'static)>>;

pub struct Gpt2Tokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl Gpt2Tokenizer {
    pub fn new() -> Result<Self> {
        //let mut tokenizer = tokenizers::Tokenizer::from_pretrained("gpt2", None)?;
        let tokenizer = tokenizers::Tokenizer::from_file("tokenizer.json")?;
        //tokenizer.add_special_tokens(&construct_special_tokens());

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

pub const LANGUAGES: [&str; 98] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "ln", "ha", "ba",
    "jw", "su",
];

use strum_macros::EnumIter;

#[derive(Debug, Copy, Clone, EnumIter)]
pub enum Language {
    English,
    Chinese,
    German,
    Spanish,
    Russian,
    Korean, 
    French,
    Japanese,
    Portuguese,
    Turkish,
    Polish,
    Catalan,
    Dutch,
    Arabic,
    Swedish,
    Italian,
    Indonesian, 
    Hindi,
    Finnish,
    Vietnamese,
    Hebrew,
    Ukrainian,
    Greek,
    Malay,
    Czech,
    Romanian,
    Danish,
    Hungarian,
    Tamil,
    Norwegian,
    Thai,
    Urdu,
    Croatian,
    Bulgarian,
    Lithuanian,
    Latin,
    Maori,
    Malayalam,
    Welsh,
    Slovak,
    Telugu,
    Persian,
    Latvian,
    Bengali,
    Serbian,
    Azerbaijani,
    Slovenian,
    Kannada,
    Estonian,
    Macedonian,
    Breton,
    Basque,
    Icelandic,
    Armenian,
    Nepali,
    Mongolian,
    Bosnian,
    Kazakh,
    Albanian,
    Swahili,
    Galician,
    Marathi,
    Punjabi,
    Sinhala,
    Khmer,
    Shona,
    Yoruba,
    Somali,
    Afrikaans,
    Occitan,
    Georgian,
    Belarusian,
    Tajik,
    Sindhi,
    Gujarati,
    Amharic,
    Yiddish,
    Lao,
    Uzbek, 
    Faroese,
    HaitianCreole,
    Pashto,
    Turkmen, 
    Nynorsk,
    Maltese,
    Samoan,
    Luxembourgish,
    Burmese,
    Bodo,
    Tagalog,
    Malagasy,
    Tatar,
    Lingala,
    Hausa,
    Bashkir,
    Javanese,
    Sundanese
}

impl Language {
    pub fn as_str(&self) -> &str {
        match self {
            Language::English => "en",
            Language::Chinese => "zh",
            Language::German => "de",
            Language::Spanish => "es",
            Language::Russian => "ru",
            Language::Korean => "ko",
            Language::French => "fr",
            Language::Japanese => "ja",
            Language::Portuguese => "pt",
            Language::Turkish => "tr",
            Language::Polish => "pl",
            Language::Catalan => "ca",
            Language::Dutch => "nl",
            Language::Arabic => "ar",
            Language::Swedish => "sv",
            Language::Italian => "it",
            Language::Indonesian => "id",
            Language::Hindi => "hi",
            Language::Finnish => "fi",
            Language::Vietnamese => "vi",
            Language::Hebrew => "he",
            Language::Ukrainian => "uk",
            Language::Greek => "el",
            Language::Malay => "ms",
            Language::Czech => "cs",
            Language::Romanian => "ro",
            Language::Danish => "da",
            Language::Hungarian => "hu",
            Language::Tamil => "ta",
            Language::Norwegian => "no",
            Language::Thai => "th",
            Language::Urdu => "ur",
            Language::Croatian => "hr",
            Language::Bulgarian => "bg",
            Language::Lithuanian => "lt",
            Language::Latin => "la",
            Language::Maori => "mi",
            Language::Malayalam => "ml",
            Language::Welsh => "cy",
            Language::Slovak => "sk",
            Language::Telugu => "te",
            Language::Persian => "fa",
            Language::Latvian => "lv",
            Language::Bengali => "bn",
            Language::Serbian => "sr",
            Language::Azerbaijani => "az",
            Language::Slovenian => "sl",
            Language::Kannada => "kn",
            Language::Estonian => "et",
            Language::Macedonian => "mk",
            Language::Breton => "br",
            Language::Basque => "eu",
            Language::Icelandic => "is",
            Language::Armenian => "hy",
            Language::Nepali => "ne",
            Language::Mongolian => "mn",
            Language::Bosnian => "bs",
            Language::Kazakh => "kk",
            Language::Albanian => "sq",
            Language::Swahili => "sw",
            Language::Galician => "gl",
            Language::Marathi => "mr",
            Language::Punjabi => "pa",
            Language::Sinhala => "si",
            Language::Khmer => "km",
            Language::Shona => "sn",
            Language::Yoruba => "yo",
            Language::Somali => "so",
            Language::Afrikaans => "af",
            Language::Occitan => "oc",
            Language::Georgian => "ka",
            Language::Belarusian => "be",
            Language::Tajik => "tg",
            Language::Sindhi => "sd",
            Language::Gujarati => "gu",
            Language::Amharic => "am",
            Language::Yiddish => "yi",
            Language::Lao => "lo",
            Language::Uzbek => "uz",
            Language::Faroese => "fo",
            Language::HaitianCreole => "ht",
            Language::Pashto => "ps",
            Language::Turkmen => "tk",
            Language::Nynorsk => "nn",
            Language::Maltese => "mt",
            Language::Samoan => "sm",
            Language::Luxembourgish => "lb",
            Language::Burmese => "my",
            Language::Bodo => "brx",
            Language::Tagalog => "tl",
            Language::Malagasy => "mg",
            Language::Tatar => "tt",
            Language::Lingala => "ln",
            Language::Hausa => "ha",
            Language::Bashkir => "ba",
            Language::Javanese => "jw",
            Language::Sundanese => "su",
        }
    }
}

pub enum SpecialToken {
    EndofText,
    StartofTranscript,
    Translate,
    Transcribe,
    StartofLM,
    StartofPrev,
    NoSpeech,
    NoTimeStamps,
    Language(Language),
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
            SpecialToken::Language(lang) => format!("<|{}|>", lang.as_str()),
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
