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
