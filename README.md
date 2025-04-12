# 🧠 Liveness Classifier for Face Anti-Spoofing

This project presents an AI-based system to detect face spoofing attacks in eKYC and access control systems. The goal is to classify input face images as either **"normal" (live)** or **"spoof" (fake)** with a liveness score ∈ [0,1].

---

## 📁 Dataset Structure

The dataset is organized into two main folders: `train` and `test`, each containing:

train/ ├── normal/ └── spoof/ test/ ├── normal/ └── spoof/


- Images are named using the format `Oi_Ij.jpg`, where `Oi` is the person ID and `Ij` is the j-th image of that person.
- Most subjects have 4 images; for those with fewer, the first image is repeated.
- Some images are flipped or rotated; those are treated as outliers or augmented with rotation-based transforms.

---

## 🛠️ Preprocessing

### Single-Image Approach:
- Resize to 224x224
- Normalize using ImageNet stats
- Apply augmentations: `HorizontalFlip`, `RandomRotation`

### Multi-Image Approach:
- Custom PyTorch `Dataset` groups 4 images per subject
- Each sample is `[4, 3, 224, 224]`

---

## 🚀 Models Implemented

### ✅ **1. ResNet50 – Single Image**
- Pretrained ResNet50 backbone
- Replace FC layer with MLP classifier
- Only fine-tune the new MLP head
- **Loss**: CrossEntropyLoss | **Optimizer**: Adam (1e-4) | **Epochs**: 10

---

### ✅ **2. ResNet50 + LSTM – Multi Image**
- Use ResNet50 to extract features for 4 images
- LSTM processes feature sequence → classify
- Entire model is trained end-to-end
- **Loss**: CrossEntropyLoss | **Optimizer**: Adam (1e-4) | **Epochs**: 5

---

### ✅ **3. ViT + Mean Pooling – Multi Image**
- Use `vit_base_patch16_224` from `timm`
- Remove original classification head
- Extract embeddings for 4 images → mean pooling → classify
- Lightweight alternative to LSTM
- **Loss**: CrossEntropyLoss | **Optimizer**: Adam (2e-5) | **Epochs**: 2

---

### ✅ **4. AutoEncoder + ResNet18 – Multi Image**
- Use AutoEncoder to reconstruct images
- Pass reconstructed images to a ResNet18 classifier
- Trained end-to-end
- **Loss**: CrossEntropyLoss | **Optimizer**: Adam (5e-4) | **Epochs**: 10

---

### ✅ **5. AutoEncoder + Reconstruction Error – One-Class Learning**
- Train AE on **normal** class only
- During inference, calculate reconstruction error
- Use ROC-AUC to find threshold to classify spoof
- **Loss**: MSELoss | **Optimizer**: Adam (1e-3) | **Epochs**: 10

---

## 📊 Evaluation

- **Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Best models**:  
  - ResNet50 + LSTM: Accuracy ~94%  
  - ViT + Mean Pooling: Accuracy ~95%  
- **AE + Reconstruction Error** is promising for open-set or real-world spoof types.

---

## 📉 Model Limitations

- Low-resolution or blurry images can confuse the model
- Dataset lacks advanced spoof types (e.g., 3D masks, deepfakes)
- Subjects wearing masks → fewer facial features
- Still images only; lacks temporal motion cues

---

## 🌐 Proposed Improvements

- Use video input to capture micro-movements
- Integrate contrastive or triplet loss
- Incorporate metadata (e.g., IR, depth)
- Use ensemble methods (ResNet + ViT)
- Generate spoof data with GANs for better generalization
- Combine biometric & contextual signals (e.g., time, location)

---

## 🔐 Multi-layer Fraud Detection System

Designed a **robust fraud detection pipeline** with 8 layers:

1. **Account Verification**  
2. **Personalized Security Questions**  
3. **Liveness Detection (ViT/CNN/AE)**  
4. **Biometric Face Verification**  
5. **Behavioral Analysis** (e.g., eye blink, head turn)  
6. **Device & Metadata Check** (IP, OS, Camera)  
7. **Anomaly Detection** (Isolation Forest, One-Class SVM)  
8. **Decision Layer** (Auto-accept, Flag, Reject)

---

## ⚙️ How to Run

1. Upload `Dataset.zip` to Google Drive
2. Open `Liveness_Classification_Resnet50Multi.ipynb` in Google Colab
3. Set up GPU runtime
4. Run all cells to train or load existing model weights
5. Use `evaluate()` to validate on dev set
6. Use `predict()` with new images to get liveness scores

---

## 📎 Repository

🔗 GitHub: [https://github.com/ShouyiLeee/TakeHomeTest](https://github.com/ShouyiLeee/TakeHomeTest)

---

## 👨‍💻 Author

- Name: [Your Full Name]  
- Email: [Your Email Address]  
- Position: Applicant for [Python Developer / AI Intern / etc.]

