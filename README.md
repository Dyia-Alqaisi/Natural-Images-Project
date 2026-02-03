# Natural Images Classification with MobileNetV2 (TensorFlow/Keras)

This project builds an end-to-end image classification workflow for the **Natural Images dataset** (8 classes). It utilizes Transfer Learning with **MobileNetV2** to achieve high accuracy and includes a robust preprocessing pipeline to handle real-world image variances.

The 8 classes are:
`airplane`, `car`, `cat`, `dog`, `flower`, `fruit`, `motorbike`, `person`

---

## 🚀 What This Project Does

The workflow is broken down into three key stages:

### 1. Image Preprocessing (256×256)
A robust pipeline that standardizes raw data before it touches the model.
* ✅ **Fixes Orientation:** Automatically corrects EXIF orientation (prevents upside-down/rotated images).
* ✅ **Consistent Format:** Converts all inputs to RGB (handles Grayscale and RGBA transparency issues).
* ✅ **True Padding:** Resizes using high-quality downsampling (LANCZOS) and adds black padding to maintain the original aspect ratio (no stretching).
* ✅ **Structured Output:** Saves processed images into a clean folder structure ready for training.

### 2. Transfer Learning with MobileNetV2
Trains a classifier using TensorFlow/Keras.
* **Data Loading:** Uses `image_dataset_from_directory` with a 70% Train / 15% Val / 15% Test split.
* **Optimization:** Implements `cache()` and `prefetch()` for faster training speed.
* **Augmentation:** Applies MobileNetV2 input scaling (`[-1, 1]`) and prevents overfitting with:
    * Random horizontal flip
    * Small random rotation
    * Small random zoom
* **Architecture:** Uses a frozen MobileNetV2 backbone (ImageNet weights) + a custom classifier head:
    * `GlobalAveragePooling`
    * `Dropout`
    * `Dense` layer(s)
    * `Softmax` output for 8 classes
* **Output:** Trains for 10 epochs, plots accuracy/loss curves, evaluates on test set, and saves the model as `MobileNetV2_classifier.keras`.

### 3. Single Image Prediction
A deployment-ready script for inference.
* Loads the saved `.keras` model.
* Applies the **exact same preprocessing** used during training (EXIF fix + padding resize).
* Predicts the class name + confidence score and displays the image with the result.

---

## 📂 Notebook Files

| File | Description |
| :--- | :--- |
| `01_data_preprocessing_natural_images.ipynb` | Preprocess raw data & export to `processed_images/` |
| `02_model_training_mobilenetv2.ipynb` | Train, evaluate, and save the `MobileNetV2_classifier` |
| `03_single_image_prediction.ipynb` | Load the saved model & predict on a new image |

---

## 🛠️ Requirements

* Python 3
* TensorFlow / Keras
* Pillow (PIL)
* NumPy
* Matplotlib

You can install dependencies via pip:
```bash
pip install tensorflow pillow numpy matplotlib
