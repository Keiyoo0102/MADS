# Global Classification of Mars-Analog Sand Dunes using Vision-Language Models

<div align="center">

[![Dataset](https://img.shields.io/badge/Dataset-Figshare-green)](https://doi.org/10.6084/m9.figshare.31168627)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

## 📖 Introduction

This repository contains the official implementation of the paper: **"Global Classification of Mars-Analog Sand Dunes using Vision-Language Models"**.

We propose a novel **Knowledge-Driven Vision-Language Framework** for high-resolution semantic segmentation of aeolian landforms on Earth, which serve as critical analogs for Martian geology. Unlike traditional RGB-based methods, our approach leverages:

1.  **Foundation Model Embeddings**: 64-dimensional feature vectors from Google Satellite Embeddings (V1), capturing rich spectral and spatial contexts.
2.  **Expert Knowledge**: Textual descriptions of dune morphologies integrated via a cross-attention mechanism to guide the segmentation process.

Our model achieves state-of-the-art performance (**OA: 96.11%, mIoU: 80.36%**) across five major Mars-analog regions: *Namib, Atacama, Mojave, Qaidam, and Yilgarn*.



## 📂 Project Structure

The repository is organized as follows:

```text
Mars-Dune-Segmentation/
├── checkpoints/           # Directory for saving model weights
├── data/                  # Dataset directory
│   ├── text_embeds_detailed.npy  # Pre-computed text embeddings (REQUIRED)
│   ├── img_dir/           # 64-channel .npy image files
│   └── ann_dir/           # Ground truth masks (.png)
├── models/                # Model architecture definitions
│   ├── fusion.py          # Our Vision-Language Fusion Module
│   └── segformer.py       # Backbone definition (SegFormer)
├── utils/                 # Utility scripts (Dataset, Metrics)
├── inference.py           # Single image inference & evaluation script
├── inference_tile.py      # Sliding-window inference for huge GeoTIFFs
├── train.py               # Main training script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

```

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone [https://github.com/Keiyoo0102/MADS.git](https://github.com/Keiyoo0102/MADS.git)
cd Mars-Dune-Segmentation
```

### 2. Set up the environment

We recommend using **Anaconda** to manage dependencies.

```bash
# Create a new environment
conda create -n dune_seg python=3.9
conda activate dune_seg

# Install PyTorch (Adjust CUDA version according to your hardware)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other requirements
pip install -r requirements.txt

```

**Key Dependencies:**

* `torch>=1.10.0`
* `rasterio` (for GeoTIFF processing)
* `numpy`, `pandas`, `tqdm`, `pillow`, `scikit-learn`

## 🌍 Data Preparation

### 1. Satellite Embeddings (GEE Source)

Our model utilizes **64-dimensional feature vectors** derived from the Google Earth Engine (GEE) Foundation Model, rather than standard RGB imagery.

* **GEE Collection ID:** `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
* **Resolution:** 10m 

You can acquire the data using the GEE JavaScript Code Editor or Python API. Below is an example snippet:

```javascript
// GEE JavaScript Code Editor Example
var roi = ee.Geometry.Rectangle([Long_min, Lat_min, Long_max, Lat_max]);
var dataset = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
                .filterDate('2023-01-01', '2023-12-31')
                .filterBounds(roi);

// Get the 64-band embedding image
var embedding = dataset.first().clip(roi);

// Export to Drive as GeoTIFF
Export.image.toDrive({
  image: embedding,
  description: 'Mars_Analog_Embedding',
  scale: 10,
  region: roi,
  fileFormat: 'GeoTIFF'
});

```

**Preprocessing Note:** The exported GeoTIFFs must be converted into `.npy` format (Shape: `64 x H x W`) for the dataloader.

### 2. Expert Knowledge (Text Embeddings)

The file `data/text_embeds_detailed.npy` contains the pre-computed text embeddings for the 10 dune categories (e.g., Crescentic, Linear, Star Dunes). **Do not delete or modify this file**, as it is essential for the cross-attention module.

### 3. Directory Layout

Please organize your training data as follows:

```text
Training_Dataset_Final/
├── img_dir/
│   ├── train/  # Contains .npy files (64, 512, 512)
│   ├── val/
│   └── test/
└── ann_dir/
    ├── train/  # Contains .png label files (512, 512)
    ├── val/
    └── test/

```

## 🚀 Usage

### 1. Training

To train the model from scratch using the fusion of Vision + Text:

```bash
python train.py \
  --data_root ./Training_Dataset_Final \
  --save_dir ./checkpoints/Ours_Model \
  --batch_size 4 \
  --epochs 50 \
  --lr 6e-5

```

### 2. Evaluation / Inference

To evaluate a trained model on the test set and calculate metrics (OA, mIoU):

```bash
python inference.py \
  --data_root ./Training_Dataset_Final \
  --weights ./checkpoints/Ours_Model/best_model.pth \
  --split test

```

### 3. Large-Scale Mapping (Huge TIF)

To perform sliding-window inference on a large GeoTIFF (e.g., the entire Namib Desert):

```bash
python inference_tile.py \
  --input_tif ./raw_data/Namib_Embedding.tif \
  --output_tif ./results/Namib_Classification.tif \
  --weights ./checkpoints/best_model.pth

```

## 🏆 Results

The final classification maps and the trained model weights are available on **Figshare**.

| Method | Backbone | Modality | OA (%) | Kappa | Download |
| --- | --- | --- | --- | --- | --- |
| **Ours** | SegFormer | Embed + Text | **96.11** | **0.9418** | [Link](https://doi.org/10.6084/m9.figshare.31168627) |




## 📄 License

This project is released under the [MIT License](https://www.google.com/search?q=LICENSE).

## 🙏 Acknowledgements

* We thank Google Earth Engine for providing the [Satellite Embeddings](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL).
* This work is built upon [SegFormer](https://github.com/NVlabs/SegFormer).

