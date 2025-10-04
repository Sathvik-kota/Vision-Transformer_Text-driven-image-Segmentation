# ViT-CIFAR10: Two-Stage W&B Sweeps (Colab)

Clean Vision Transformer implementation for CIFAR-10 with two-stage hyperparameter optimization using Weights & Biases sweeps. Designed for Google Colab with Drive persistence.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg) ![W&B](https://img.shields.io/badge/Weights&Biases-Sweeps-yellow)

## Features

- **Pure PyTorch ViT**: PatchEmbed, pre-norm blocks, DropPath, GELU MLP
- **Robust Training**: AMP, MixUp, RandAugment, cosine+warmup, early stopping  
- **Two-Stage Optimization**: Fast sweep on proxy subset → full training on best configs
- **Drive Integration**: Auto-save results, configs, and checkpoints to Google Drive

## Quick Setup (Colab)

```python
from pathlib import Path
from google.colab import drive

# Mount Drive & create directories
drive.mount('/content/drive')
SWEEP_SAVE_DIR = Path("/content/drive/MyDrive/vit_cifar10_sweep_results")
FULL_SAVE_DIR = Path("/content/drive/MyDrive/vit_cifar10_fulltrain_results")
for p in [SWEEP_SAVE_DIR, FULL_SAVE_DIR]:
    p.mkdir(parents=True, exist_ok=True)
```

## Stage 1: W&B Bayesian Sweep (20% subset, 20 runs)

```python
import wandb

# Define sweep configuration
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "patch_size": {"values": [2,4]},
        "embed_dim": {"values": [128,192]}, 
        "depth": {"values": [4,6]},
        "num_heads": {"values": [2,4]},
        "lr": {"min":1e-5, "max":1e-3},
        "optimizer": {"values": ["adamw","sgd"]},
        "batch_size": {"values":[64,128]},
        "mixup_alpha": {"values":[0.0,0.1]},
        "subset_ratio": {"value": 0.2}  # 20% proxy dataset
    }
}

# Launch sweep
sweep_id = wandb.sweep(sweep_config, project="vit-cifar10-sweep")
wandb.agent(sweep_id, function=agent_fn, count=20)
```

## Stage 2: Full Training on Top-3 Configs

```python
# Auto-select top 3 configs from Stage 1 results
top_configs = load_top_configs_from_drive(SWEEP_SAVE_DIR, top_k=3)

# Full training with extended epochs
for cfg in top_configs:
    cfg.update({"subset_ratio": 1.0, "epochs": 120, "patience": 15})
    model, best_acc = run_fulltrain(cfg, save_dir=FULL_SAVE_DIR)
    
    # Test evaluation with confusion matrix
    test_acc, cm = evaluate_model(model, test_loader, device, class_names=CIFAR10_CLASSES)
```

## Model Architecture

```python
class VisionTransformer(nn.Module):
    # Compact ViT for CIFAR-10 (32×32)
    # - PatchEmbed: Conv2d(kernel=stride=patch_size) 
    # - Learnable [CLS] token + positional embeddings
    # - Pre-norm Transformer blocks with DropPath
    # - Classification head
```

## Training Stack

- **Data**: Stratified 90/10 split, RandAugment, standard CIFAR-10 normalization
- **Optimization**: AdamW, cosine LR with warmup, AMP mixed precision
- **Regularization**: MixUp, DropPath, gradient clipping, early stopping
- **Monitoring**: W&B logging, best model checkpointing

## Best Config Found

```json
{
  "embed_dim": 128, "depth": 6, "num_heads": 4, "patch_size": 4,
  "lr": 0.000584, "optimizer": "adamw", "scheduler": "cosine",
  "batch_size": 128, "drop_path_rate": 0.1, "use_randaugment": true
}
```

Achieved 87.22% validation accuracy on CIFAR-10.

## File Structure

```
/content/drive/MyDrive/
├── vit_cifar10_sweep_results/     # Stage 1 sweep JSONs/checkpoints
├── vit_cifar10_fulltrain_results/ # Stage 2 full training results  
└── vit_cifar10_fulltrain_top3.json # Top configs for Stage 2
```

## Usage

1. Run all code cells in sequence
2. Stage 1 automatically saves top configs to Drive
3. Stage 2 loads configs and runs full training
4. Results include test accuracy + confusion matrix

## Requirements

PyTorch 2.0+, torchvision, scikit-learn, wandb, seaborn, matplotlib

**Note**: Requires Colab GPU runtime for reasonable training speed.
# Vision Transformer on CIFAR-10 - q1.ipnyb
## 📌 Overview  
This project trains and evaluates deep learning models on the **CIFAR-10 dataset** using a **two-stage training pipeline**:  

- **Stage 1: Hyperparameter Sweep**  
  - Train on **45,000 samples**  
  - Validate on **5,000 samples**  
  - Run **W&B sweeps** to explore multiple configurations  
  - Select the **Top-3 configs** based on validation accuracy  

- **Stage 2: Final Training & Evaluation**  
  - Retrain top configs on the **entire training set (50,000 samples)**  
  - Apply **early stopping** (patience = 15, max epochs = 120)  
  - Evaluate final models on the **10,000 test samples**  
  - Compare validation vs test performance  
## ⚙️ Training Strategy  

### Stage 1: Hyperparameter Search  
- **Data split:**  
  - 45,000 train  
  - 5,000 validation  
- **Transforms:** RandAugment + RandomCrop + RandomHorizontalFlip + Normalization  
- Models trained with multiple configs (optimizers, learning rates, dropout, etc.)  
- **Top-3 configs saved** based on validation accuracy  

### Stage 2: Full Training & Test Evaluation  
- **Data split:**  
  - 50,000 train (with 10% held out for validation during training)  
  - 10,000 test (never used in training or sweeps)  
- Training: up to **120 epochs**  
- **Early stopping:** patience = 15  
- **Outputs:** Best checkpoints stored in Google Drive  
- Final evaluation with **test accuracy, confusion matrix, and classification report**
### 🔧 Stage 1: Hyperparameter Search Space  

The following hyperparameters were tuned using **W&B Sweeps** (Bayesian optimization):  

- **Model Architecture**  
  - `patch_size`: [2, 4]  
  - `embed_dim`: [128, 192]  
  - `depth`: [4, 6]  
  - `num_heads`: [2, 4]  

- **Training Parameters**  
  - `lr`: [1e-5 → 1e-3] (log-scaled search)  
  - `optimizer`: ["adamw", "sgd"]  
  - `scheduler`: ["cosine", "step"]  
  - `batch_size`: [64, 128]  
  - `mixup_alpha`: [0.0, 0.1]  

- **Fixed Parameters (Stage-1)**  
  - `epochs`: 20  
  - `patience`: 6 (early stopping)  
  - `subset_ratio`: 0.2 (proxy dataset, ~20% of training data used)  

**Goal:** Maximize `val_acc` on a held-out 10% validation split.  
### 🏆 Best Combination (Stage-1)

- `patch_size`: 4  
- `embed_dim`: 128  
- `depth`: 6  
- `num_heads`: 4  
- `lr`: 0.0005836  
- `optimizer`: adamw  
- `scheduler`: cosine  
- `batch_size`: 128  
- `mixup_alpha`: 0.0  

## 📊 Results & Visualization  

- **Stage 1 Plots:**  
  - Training and Validation curves 

- **Stage 2 Plots:**  
  - Training and validation metrics (loss and accuracy) per model
  - Test Accuracy, precesion , reall , f1  per model
  - Confusion Matrix
  - Training and Validatin plots ( loss and accuracy curves )


# Text-Driven Video Segmentation with CLIPSeg and SAM2

A powerful computer vision pipeline that combines **CLIPSeg** and **SAM2** for high-quality text-driven image and video segmentation. This project enables users to segment and track objects in images and videos using natural language descriptions.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.9+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **Text-Driven Segmentation**: Segment objects using natural language descriptions
- **Multi-Object Tracking**: Track multiple objects simultaneously in videos
- **High-Quality Masks**: Combines CLIPSeg's text understanding with SAM2's precise segmentation
- **Interactive Interface**: User-friendly interactive modes for both images and videos
- **GPU Accelerated**: Optimized for CUDA-enabled environments
- **Multiple Input Formats**: Supports local files, URLs, and various image/video formats

## 🏗️ Architecture

The pipeline consists of three main stages:

1. **Text-to-Region Detection (CLIPSeg)**: Generates initial heatmaps based on text prompts
2. **Bounding Box Extraction**: Converts heatmaps to region proposals
3. **High-Quality Segmentation (SAM2)**: Refines masks with precise boundaries

```
Text Prompt → CLIPSeg → Heatmap → Bounding Boxes → SAM2 → Refined Masks
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Google Colab or local environment with GPU support

### Installation

```bash
# Clone repositories
git clone https://github.com/facebookresearch/segment-anything-2.git
git clone https://github.com/IDEA-Research/GroundingDINO.git

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python matplotlib pillow requests numpy tqdm wget
pip install transformers accelerate supervision addict yacs timm einops
pip install hydra-core omegaconf yaspin yapf iopath fvcore fairscale
pip install git+https://github.com/timojl/clipseg.git
pip install yt-dlp imageio imageio-ffmpeg

# Install SAM2
cd segment-anything-2 && pip install -e . && cd ..
```

### Download Model Checkpoints

```python
import os
import wget

os.makedirs("checkpoints", exist_ok=True)

# Download SAM2 checkpoint
wget.download(
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    "checkpoints/sam2.1_hiera_large.pt"
)
```

## 📖 Usage

### Image Segmentation

```python
from text_segmenter import TextDrivenSegmenter

# Initialize the segmenter
segmenter = TextDrivenSegmenter()

# Segment objects in an image
results = segmenter.process_image(
    image_input="path/to/image.jpg",  # or URL
    text_prompt="person . dog . car",  # separate objects with periods
    clip_threshold=0.5
)

# Visualize results
visualize_results(results, "person . dog . car")
```

### Video Object Tracking

```python
from video_tracker import VideoObjectTracker

# Initialize tracker
segmenter = TextDrivenSegmenter()
tracker = VideoObjectTracker(segmenter)

# Process video
output_mp4, output_gif = tracker.process_video(
    video_input="path/to/video.mp4",  # or URL
    text_prompt="horse",
    max_frames=150
)
```

### Interactive Mode

```python
# Run interactive image segmentation
interactive_segmentation()

# Run interactive video segmentation  
run_interactive_video_segmentation()
```

## 🎯 Examples

### Single Object Segmentation
```python
segment_with_text_prompt(
    "https://images.unsplash.com/photo-1601758228041-f3b2795255f1?w=800",
    "dog"
)
```

### Multi-Object Segmentation
```python
segment_with_text_prompt(
    "street_scene.jpg",
    "car . person . bicycle . traffic light"
)
```

### Video Tracking
```python
tracker.process_video(
    "https://www.pexels.com/video/cars-on-highway-1234567/",
    "car"
)
```

## ⚙️ Configuration

### Key Parameters

- **`clip_threshold`**: Controls CLIPSeg sensitivity (0.0-1.0, default: 0.5)
- **`min_area_ratio`**: Minimum object size ratio (default: 0.001)
- **`max_frames`**: Maximum frames to process for videos (default: 150)

### Performance Tuning

```python
# For better accuracy on small objects
results = segmenter.process_image(
    image_input="image.jpg",
    text_prompt="small object",
    clip_threshold=0.3  # Lower threshold for better recall
)

# For faster processing
tracker.process_video(
    video_input="video.mp4",
    text_prompt="object",
    max_frames=50  # Process fewer frames
)
```

## 🔬 Technical Details

### Model Components

- **CLIPSeg**: Text-image segmentation model for initial object localization
- **SAM2**: Segment Anything Model 2 for high-quality mask refinement
- **Pipeline**: Custom integration optimizing both models' strengths

### System Requirements

- **GPU Memory**: 8GB+ VRAM recommended
- **RAM**: 16GB+ system memory
- **Storage**: 5GB+ for model checkpoints
- **CUDA**: Version 11.8 or higher

### Performance Metrics

- **Image Processing**: ~2-5 seconds per image (depends on resolution and object count)
- **Video Processing**: ~0.5-1.5 seconds per frame
- **Memory Usage**: ~6-8GB GPU memory during inference

## 📊 Supported Formats

### Input Formats
- **Images**: JPEG, PNG, TIFF, BMP, WebP
- **Videos**: MP4, AVI, MOV, MKV, WebM
- **Sources**: Local files, HTTP URLs, Google Drive paths

### Output Formats
- **Images**: PNG with transparency support
- **Videos**: MP4 (H.264) and GIF animations
- **Masks**: NumPy arrays and binary masks

## 🎨 Visualization Features

- **Color-coded masks** for multi-object segmentation
- **Bounding box overlays** with object labels
- **Interactive matplotlib displays**
- **Progress tracking** with tqdm progress bars
- **HTML video players** for Colab environments

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size or image resolution
   img = cv2.resize(img, (800, 600))
   ```

2. **Model Download Failures**
   ```bash
   # Manual download
   wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
   ```

3. **CLIPSeg Installation Issues**
   ```bash
   pip install --no-cache-dir git+https://github.com/timojl/clipseg.git
   ```

### Performance Optimization

- Use **GPU** when available (automatically detected)
- **Resize large images** before processing
- **Limit video length** for faster processing
- **Adjust CLIPSeg threshold** based on object characteristics

## 📚 References

- [SAM2: Segment Anything in Images and Videos](https://github.com/facebookresearch/segment-anything-2)
- [CLIPSeg: Image Segmentation Using Text and Image Prompts](https://github.com/timojl/clipseg)
- [CLIP: Learning Transferable Visual Representations](https://github.com/openai/CLIP)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Facebook Research for SAM2
- OpenAI for CLIP and CLIPSeg innovations
- The open-source computer vision community

## 📧 Contact

- **Issues**: Please use the GitHub Issues tab
- **Discussions**: GitHub Discussions for questions and ideas
- **Updates**: Watch this repository for the latest developments

---

**⭐ Star this repository if you find it useful!**

**🔗 Share with the community and help advance text-driven computer vision!**
