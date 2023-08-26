# Whisper Burn: Rust Implementation of OpenAI's Whisper Transcription Model

**Whisper Burn** is a Rust implementation of OpenAI's Whisper transcription model using the Rust deep learning framework, Burn.

## License

This project is licensed under the terms of the MIT license.

## Model Files

The OpenAI Whisper models that have been converted to work in burn are available in the whisper-burn space on Hugging Face. You can find them at [https://huggingface.co/Gadersd/whisper-burn](https://huggingface.co/Gadersd/whisper-burn).

If you have a custom fine-tuned model you can easily convert it to burn's format. Here is an example of converting OpenAI's tiny en model. The tinygrad dependency of the dump.py script should be installed from source not with pip.

```
cd python
wget https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt
python3 dump.py tiny.en.pt tiny_en
mv tiny_en ../
cd ../
cargo run --release --bin convert tiny_en
```

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

The audio file must be have a sample rate of 16k and be single-channel.

```
sox audio.wav -r 16000 -c 1 audio16k.wav
```
Now transcribe.

```
# with tch backend (default)
cargo run --release --bin transcribe tiny_en audio16k.wav transcription.txt

# or with wgpu backend (may be unstable for large models)
cargo run --release --features wgpu-backend --bin transcribe tiny_en audio16k.wav transcription.txt
```

This usage assumes that "audio16k.wav" is the audio file you want to transcribe, and "tiny_en" is the model to use. Please adjust according to your specific needs.

Enjoy using **Whisper Burn**!