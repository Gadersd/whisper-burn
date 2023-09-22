# Portions of code adapted from the tinygrad project 
# (https://github.com/tinygrad/tinygrad) under the MIT License.

import sys
import pathlib
import base64
import multiprocessing
import numpy as np
from typing import Optional
from extra.utils import download_file
from tinygrad.nn.state import torch_load, load_state_dict
from tinygrad.helpers import getenv
import tinygrad.nn as nn
from tinygrad.tensor import Tensor
import torch

# TODO: you have written this fifteen times
class MultiHeadAttention:
  def __init__(self, n_state, n_head):
    self.n_head = n_head
    self.query = nn.Linear(n_state, n_state)
    self.key = nn.Linear(n_state, n_state, bias=False)
    self.value = nn.Linear(n_state, n_state)
    self.out = nn.Linear(n_state, n_state)

  def __call__(self, x:Tensor, xa:Optional[Tensor]=None, mask:Optional[Tensor]=None):
    q = self.query(x)
    k = self.key(xa or x)
    v = self.value(xa or x)
    wv, qk = self.qkv_attention(q, k, v, mask)
    # NOTE: we aren't returning qk
    return self.out(wv)

  def qkv_attention(self, q, k, v, mask=None):
    n_batch, n_ctx, n_state = q.shape
    scale = (n_state // self.n_head) ** -0.25
    q = q.reshape(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
    k = k.reshape(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
    v = v.reshape(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    qk = q @ k
    if mask is not None: qk = qk + mask[:n_ctx, :n_ctx]
    w = qk.softmax(-1)
    return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()

class ResidualAttentionBlock:
  def __init__(self, n_state, n_head, cross_attention=False):
    self.attn = MultiHeadAttention(n_state, n_head)
    self.attn_ln = nn.LayerNorm(n_state)

    self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
    self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

    self.mlp = [nn.Linear(n_state, n_state*4), Tensor.gelu, nn.Linear(n_state*4, n_state)]
    self.mlp_ln = nn.LayerNorm(n_state)

  def __call__(self, x, xa=None, mask=None):
    x = x + self.attn(self.attn_ln(x), mask=mask)
    if self.cross_attn: x = x + self.cross_attn(self.cross_attn_ln(x), xa)
    x = x + self.mlp_ln(x).sequential(self.mlp)
    return x

class AudioEncoder:
  def __init__(self, n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer, **_):
    self.n_mels = n_mels
    self.n_audio_state = n_audio_state

    self.conv1 = nn.Conv1d(n_mels, n_audio_state, kernel_size=3, padding=1)
    self.conv2 = nn.Conv1d(n_audio_state, n_audio_state, kernel_size=3, stride=2, padding=1)
    self.blocks = [ResidualAttentionBlock(n_audio_state, n_audio_head) for _ in range(n_audio_layer)]
    self.ln_post = nn.LayerNorm(n_audio_state)
    self.positional_embedding = Tensor.randn(n_audio_ctx, n_audio_state)

  def __call__(self, x):
    x = self.conv1(x).gelu()
    x = self.conv2(x).gelu()
    print(x.numpy())
    x = x.permute(0, 2, 1)
    x = x + self.positional_embedding[:x.shape[1]]
    x = x.sequential(self.blocks)
    x = self.ln_post(x)
    return x

class TextDecoder:
  def __init__(self, n_vocab, n_text_ctx, n_text_state, n_text_head, n_text_layer, **_):
    self.token_embedding = nn.Embedding(n_vocab, n_text_state)
    self.positional_embedding = Tensor.randn(n_text_ctx, n_text_state)
    self.blocks = [ResidualAttentionBlock(n_text_state, n_text_head, cross_attention=True) for _ in range(n_text_layer)]
    self.ln = nn.LayerNorm(n_text_state)
    #mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)

  def __call__(self, x, xa):
    offset = 0
    x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]

    seqlen, start_pos = x.shape[1], 0

    mask = np.full((1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=np.float32)
    mask = np.triu(mask, k=start_pos + 1)  # TODO: this is hard to do in tinygrad
    mask = Tensor(mask)

    for block in self.blocks: x = block(x, xa, mask)
    x = self.ln(x)
    return x @ self.token_embedding.weight.T

class Whisper:
  def __init__(self, dims):
    self.encoder = AudioEncoder(**dims)
    self.decoder = TextDecoder(**dims)

  def __call__(self, mel:Tensor, tokens:Tensor):
    return self.decoder(tokens, self.encoder(mel))









'''
def save_scalar(s, name, path):
    np.save(pathlib.Path(path, f'{name}.npy'), s)

def save_tensor(tensor, name, path):
    np.save(pathlib.Path(path, f'{name}.npy'), tensor.numpy())
'''

def save_scalar(s, name, path):
    s = np.array([1.0, float(s)]).astype(np.float32)
    np.save(pathlib.Path(path, f'{name}.npy'), s)

def save_tensor(tensor, name, path):
    tensor_numpy = tensor.numpy()
    tensor_dims = np.array(tensor_numpy.shape)
    tensor_values = tensor_numpy.flatten()
    tensor_to_save = np.concatenate((tensor_dims, tensor_values)).astype(np.float32)
    np.save(pathlib.Path(path, f'{name}.npy'), tensor_to_save)

def save_linear(linear, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_tensor(linear.weight.transpose(), 'weight', path) # PyTorch and Tinygrad strangely transpose linear weights so reverse that
    if linear.bias is not None:
        save_tensor(linear.bias, 'bias', path)

def save_layer_norm(layer_norm, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_tensor(layer_norm.weight, 'weight', path)
    save_tensor(layer_norm.bias, 'bias', path)
    save_scalar(layer_norm.eps, 'eps', path)

def save_mlp(mlp1, mlp2, path):
    save_linear(mlp1, pathlib.Path(path, 'mlp1'))
    save_linear(mlp2, pathlib.Path(path, 'mlp2'))

def save_conv1d(conv1d, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_tensor(conv1d.weight, 'weight', path)
    if conv1d.bias is not None:
        save_tensor(conv1d.bias, 'bias', path)

def save_embedding(embedding, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_tensor(embedding.weight, 'weight', path)


def save_multihead_attention(multihead_attention, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_linear(multihead_attention.query, pathlib.Path(path, 'query'))
    save_linear(multihead_attention.key, pathlib.Path(path, 'key'))
    save_linear(multihead_attention.value, pathlib.Path(path, 'value'))
    save_linear(multihead_attention.out, pathlib.Path(path, 'out'))
    save_scalar(multihead_attention.n_head, "n_head", path)

def save_residual_attention_block(residual_attention_block, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_multihead_attention(residual_attention_block.attn, pathlib.Path(path, 'attn'))
    save_layer_norm(residual_attention_block.attn_ln, pathlib.Path(path, 'attn_ln'))
    if residual_attention_block.cross_attn is not None:
        save_multihead_attention(residual_attention_block.cross_attn, pathlib.Path(path, 'cross_attn'))
        save_layer_norm(residual_attention_block.cross_attn_ln, pathlib.Path(path, 'cross_attn_ln'))
    save_mlp(residual_attention_block.mlp[0], residual_attention_block.mlp[2], pathlib.Path(path, 'mlp'))
    save_layer_norm(residual_attention_block.mlp_ln, pathlib.Path(path, 'mlp_ln'))

def save_audio_encoder(audio_encoder, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_conv1d(audio_encoder.conv1, pathlib.Path(path, 'conv1'))
    save_conv1d(audio_encoder.conv2, pathlib.Path(path, 'conv2'))
    for i, block in enumerate(audio_encoder.blocks):
        save_residual_attention_block(block, pathlib.Path(path, f'block_{i}'))
    save_layer_norm(audio_encoder.ln_post, pathlib.Path(path, 'ln_post'))
    save_tensor(audio_encoder.positional_embedding, 'positional_embedding', path)
    save_scalar(len(audio_encoder.blocks), "n_layer", path)
    save_scalar(audio_encoder.n_mels, "n_mels", path)
    save_scalar(audio_encoder.n_audio_state, "n_audio_state", path)

def save_text_decoder(text_decoder, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_embedding(text_decoder.token_embedding, pathlib.Path(path, 'token_embedding'))
    save_tensor(text_decoder.positional_embedding, 'positional_embedding', path)
    for i, block in enumerate(text_decoder.blocks):
        save_residual_attention_block(block, pathlib.Path(path, f'block_{i}'))
    save_layer_norm(text_decoder.ln, pathlib.Path(path, 'ln'))
    save_scalar(len(text_decoder.blocks), "n_layer", path)

def save_whisper(whisper, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_audio_encoder(whisper.encoder, pathlib.Path(path, 'encoder'))
    save_text_decoder(whisper.decoder, pathlib.Path(path, 'decoder'))

import re

def download_and_save_whisper(model_name, save_name):
  state = torch_load(model_name)
  model = Whisper(state['dims'])
  load_state_dict(model, state['model_state_dict'])
  save_whisper(model, pathlib.Path(f'{save_name}'))


def install_whisper_files(model_name, output_name):
  # Download and save the file
  download_and_save_whisper(model_name, output_name)

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print(f"Usage: python3 {sys.argv[0]} <model_name> <output_name>")
    sys.exit(1)
  
  model_name = sys.argv[1]
  output_name = sys.argv[2]
  install_whisper_files(model_name, output_name)