from datasets import load_dataset
import json

DATASET_ID = "roneneldan/TinyStories"
OUTPUT_FILE_PATH = "/home/bsc/bsc082280/data/roneneldan_TinyStories/original_data/train.jsonl"

dataset = load_dataset(
    DATASET_ID,
)

print(f"Loaded dataset: {dataset}")

print("Saving dataset to JSONL format...")

with open(OUTPUT_FILE_PATH, "w") as f:

    # TODO we are only using train split because we are doing quick testing, not real trainings
    for example in dataset["train"]:

        f.write(json.dumps({
            "text": example["text"]
        })+"\n")

print("Dataset saved to JSONL format successfully.")

    

    
    