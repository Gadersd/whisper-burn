import sys

from multiple_datasets.hub_default_utils import convert_hf_whisper 

if len(sys.argv) != 3:
    print("Usage: python3 script.py <huggingface_repo> <output_model_file>")
    sys.exit(1)
    
hf_repo = sys.argv[1]
out_file = sys.argv[2]

convert_hf_whisper(hf_repo, out_file)