import argparse
import os

from huggingface_hub import hf_hub_download


def download_models(local_dir="models"):
    repo_id = "rodrigomt/s2-pro-gguf"
    files = ["s2-pro-q4_k_m.gguf", "tokenizer.json"]

    os.makedirs(local_dir, exist_ok=True)

    for file in files:
        print(f"Downloading {file} to {local_dir}...")
        hf_hub_download(
            repo_id=repo_id, filename=file, local_dir=local_dir, local_dir_use_symlinks=False
        )
    print("Download complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Fish Speech GGUF models")
    parser.add_argument("--dir", default="models", help="Local directory to save models")
    args = parser.parse_args()
    download_models(args.dir)
