# Whisper Burn: Rust Implementation of OpenAI's Whisper Transcription Model

**Whisper Burn** is a Rust implementation of OpenAI's Whisper Transcription model using the Rust deep learning framework, Burn.


## License

This project is licensed under the terms of the MIT license.

## Installation & Usage

Before starting, ensure you have the necessary tools & libraries installed in your system. These instructions are for both CUDA and Mac users.

#### 1. Clone the Repository

Clone the repository to your local machine using the following command:

```
git clone https://github.com/Gadersd/whisper-burn.git
```

Then, navigate to the project folder:

```
cd burn
```

#### 2. Download Whisper Tiny English Model

Use the following commands to download the Whisper tiny English model:

```
wget https://huggingface.co/Gadersd/whisper-burn/blob/main/tiny_en/tiny_en.cfg
wget https://huggingface.co/Gadersd/whisper-burn/blob/main/tiny_en/tiny_en.mpk.gz
```

### CUDA USERS

#### 3. Set Environment Variable for Torch CUDA Version

Set your Torch CUDA version environment variable

```
export TORCH_CUDA_VERSION=cu113
```

#### 4. Run the Application

Once you've finished setting up, you can run the application using this command:

```
cargo run --release audio.wav tiny_en/tiny_en
```

### MAC USERS

#### 3. Run the Application

Run the application with the following command:

```
cargo run --release audio.wav tiny_en/tiny_en
```

This usage assumes that "audio.wav" is the audio file you want to transcribe, and "tiny_en/tiny_en" is the model to use. Please adjust according to your specific needs.

Enjoy using **Whisper Burn**!
