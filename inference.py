import argparse
import torch
import numpy as np
import os
from models.fusion import get_fusion_model
from utils.metrics import Metric
from utils.dataset import SandDuneDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--text_embed_path", default="./data/text_embeds_detailed.npy")
    parser.add_argument("--weights", required=True, help="Path to best_model.pth")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Data
    ds = SandDuneDataset(args.data_root, split=args.split, transform=False, text_embed_path=args.text_embed_path)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # Load Model
    model = get_fusion_model(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    model.eval()

    metric = Metric(10)
    print("Running inference...")

    with torch.no_grad():
        for imgs, texts, lbls in tqdm(loader):
            imgs, texts = imgs.to(device), texts.to(device)
            outputs = model(imgs, texts)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            metric.add_batch(preds, lbls.numpy())

    results = metric.evaluate()
    print(f"\nFinal Results on {args.split} set:")
    print(f"OA: {results['OA']:.2%}")
    print(f"mIoU: {results['mIoU']:.2%}")


if __name__ == "__main__":
    main()