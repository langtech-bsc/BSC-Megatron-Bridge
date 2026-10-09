from huggingface_hub import snapshot_download
from datasets import load_dataset
import json

MODEL_ID = "Qwen/Qwen3-8B"
DATASET_ID = "roneneldan/TinyStories"

if True:
    print("Downloading model from Hugging Face Hub...")
    snapshot_download(
        repo_id=MODEL_ID,
    )
    print("Model downloaded successfully.")

if True:
    print("Loading dataset...")

    dataset = load_dataset(
        DATASET_ID,
    )

    print("Dataset loaded successfully.")

    

    
    