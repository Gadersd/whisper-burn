# Whisper Burn: Rust Implementation of OpenAI's Whisper Transcription Model

**Whisper Burn** is a Rust implementation of OpenAI's Whisper transcription model using the Rust deep learning framework, Burn.

## License

This project is licensed under the terms of the MIT license.

## Model Files

All the Whisper models that have been converted to work in burn are available in the whisper-burn space on Hugging Face. You can find them at [https://huggingface.co/Gadersd/whisper-burn](https://huggingface.co/Gadersd/whisper-burn).

#### 1. Clone the Repository

Clone the repository to your local machine using the following command:

```
git clone https://github.com/Gadersd/whisper-burn.git
```

Then, navigate to the project folder:

```
cd whisper-burn
```

#### 2. Download Whisper Tiny English Model

Use the following commands to download the Whisper tiny English model:

```
wget https://huggingface.co/Gadersd/whisper-burn/resolve/main/tiny_en/tiny_en.cfg
wget https://huggingface.co/Gadersd/whisper-burn/resolve/main/tiny_en/tiny_en.mpk.gz
```

#### 3. Run the Application

Once you've finished setting up, you can run the application using this command:

```
cargo run --release audio.wav tiny_en
```

This usage assumes that "audio.wav" is the audio file you want to transcribe, and "tiny_en" is the model to use. Please adjust according to your specific needs.

Enjoy using **Whisper Burn**!
