import argparse
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import math
import torch.nn.functional as F
from models.fusion import get_fusion_model

NODATA_VAL = 255
BLOCK_SIZE = 2048
STRIDE = 256
INPUT_SIZE = 512


def process_chunk(img_chunk, model, text_tensor, device):
    # (Re-implement the sliding window logic from your original script here)
    # Simplified for brevity in this response, paste your original process_chunk_in_memory function logic here
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tif", required=True)
    parser.add_argument("--output_tif", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--text_embed_path", default="./data/text_embeds_detailed.npy")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Model & Resources
    model = get_fusion_model(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    model.eval()

    text_np = np.load(args.text_embed_path)
    text_tensor = torch.from_numpy(text_np).float().unsqueeze(0).to(device)

    # Process TIF (Copy the rasterio logic from your original script)
    print(f"Processing {args.input_tif} ...")
    # ... (Paste your Rasterio window loop here) ...


if __name__ == "__main__":
    main()