# PRODIGY_ML_05
# 🍕🥗 Task 5 – Food 101 Image Classification | Machine Learning Internship @ Prodigy Infotech

This project focuses on building a deep learning image classifier to identify types of food from images using the popular **Food-101** dataset. The task highlights the real-world application of computer vision in the food industry, health tech, and mobile apps.

---

## 📁 Dataset Overview

- **Source:** [ETH Zurich - Food-101 Dataset](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
- **Total Classes:** 101 different types of food
- **Images per Class:** 1,000 (750 for training, 250 for testing)
- **Image Format:** `.jpg` files
- **Total Size:** ~5 GB

The dataset contains labeled images of popular dishes, making it ideal for evaluating large-scale image classification models.

---

## 📦 Project Structure

PRODIGY_ML_05/
├── Prodigy05.ipynb # Main Jupyter notebook for training and evaluation
├── food-101/ # Root dataset directory after extraction
│ ├── images/ # Subfolders by food category
│ ├── meta/
│ │ ├── train.txt
│ │ └── test.txt
├── training_hist.json # Training history (accuracy/loss) stored as JSON
├── model/ # (Optional) Saved model checkpoints
└── README.md # Project documentation

yaml
Copy
Edit

---

## 🛠️ Requirements

- Python 3.8+
- TensorFlow / PyTorch
- NumPy
- pandas
- matplotlib
- scikit-learn
- tqdm
- Pillow

---

## 🚀 How to Run

1. **Download** the dataset and extract into the `food-101/` folder.
2. **Open** `Prodigy05.ipynb`.
3. **Run all cells** to:
   - Load and preprocess images
   - Build and compile the CNN model
   - Train on the training set and evaluate on the test set
   - Visualize training history (accuracy and loss curves)
   - Save results and model if required

---

## 📊 Model Summary

- **Architecture:** Convolutional Neural Network (CNN)
- **Input Size:** Resized images to `128x128` or `224x224` (depending on model)
- **Output:** Multiclass classification (101 food categories)
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam
- **Evaluation Metric:** Top-1 Accuracy

📈 **Sample Training Results** (10 epochs):  
- Final Train Accuracy: ~44.1%  
- Final Validation Accuracy: ~66.3%  
- Validation Loss decreased from 3.48 to 1.56 over 10 epochs  
*(See `training_hist.json` for detailed stats)*

---

## 💡 Key Learnings

- Learned how to handle large-scale image datasets efficiently
- Understood how CNNs can learn discriminative features between similar food items
- Realized the importance of data augmentation and preprocessing for image-based models
- Explored practical use cases in health tracking apps, restaurant systems, and AR-based food labeling

---

## 🔬 Limitations & Future Work

- Performance may be improved using **transfer learning** (e.g., ResNet50, EfficientNet)
- Explore **data augmentation** and **batch normalization**
- Real-time food recognition using camera input for mobile or web applications
- Use **Top-5 Accuracy** to measure model flexibility for similar dishes
