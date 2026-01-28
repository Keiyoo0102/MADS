import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import Metric
from utils.dataset import SandDuneDataset
from models.fusion import get_fusion_model


def get_args():
    parser = argparse.ArgumentParser(description="Train Mars Dune Segmentation Model")
    parser.add_argument("--data_root", type=str, required=True, help="Path to Training_Dataset_Final")
    parser.add_argument("--text_embed_path", type=str, default="./data/text_embeds_detailed.npy",
                        help="Path to text embeddings npy")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save weights")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_classes", type=int, default=10)
    return parser.parse_args()


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Start Training on {device}...")
    os.makedirs(args.save_dir, exist_ok=True)

    # Dataset
    train_ds = SandDuneDataset(args.data_root, split='train', transform=True, text_embed_path=args.text_embed_path)
    val_ds = SandDuneDataset(args.data_root, split='val', transform=False, text_embed_path=args.text_embed_path)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = get_fusion_model(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.epochs, power=1.0)
    metric = Metric(args.num_classes)

    best_miou = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for imgs, texts, lbls in pbar:
            imgs, texts, lbls = imgs.to(device), texts.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, texts)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        scheduler.step()

        # Validation
        model.eval()
        metric.reset()
        with torch.no_grad():
            for imgs, texts, lbls in tqdm(val_loader, desc="Validating"):
                imgs, texts, lbls = imgs.to(device), texts.to(device), lbls.to(device)
                outputs = model(imgs, texts)
                preds = torch.argmax(outputs, dim=1)
                metric.add_batch(preds.cpu().numpy(), lbls.cpu().numpy())

        results = metric.evaluate()
        print(f"Epoch {epoch + 1}: mIoU={results['mIoU']:.2%}, OA={results['OA']:.2%}")

        if results['mIoU'] > best_miou:
            best_miou = results['mIoU']
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print("🔥 New Best Model Saved!")


if __name__ == "__main__":
    args = get_args()
    train(args)