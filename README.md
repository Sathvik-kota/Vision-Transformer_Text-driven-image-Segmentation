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
