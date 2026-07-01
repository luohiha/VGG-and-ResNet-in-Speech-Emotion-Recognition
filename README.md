# Speech Emotion Recognition — VGG vs ResNet

> 🎓 This is my senior undergraduate **Data Science** project at Pepperdine University (CS 210).  

---

## 📁 Repository Files

| File | Description |
|------|-------------|
| `audio_to_mel_spectrogram.py` | Converts raw audio files into Mel-spectrogram images for feature extraction |
| `VGG.py` | VGG-based model implementation for speech emotion classification |
| `Resnet.ipynb` | ResNet-based model implementation with training, evaluation, and visualization |

---

## 📌 Project Overview

This project explores and compares two classic deep learning architectures — **VGG** and **ResNet** — for the task of **Speech Emotion Recognition (SER)**. Audio samples from three publicly available datasets (CREMA-D, TESS, RAVDESS) are converted into Mel-spectrograms, which are then used as inputs to train the models.

### Pipeline Overview

1. **Data Preparation** — Audio augmentation (white noise, time stretching, pitch shifting) → Mel-spectrogram generation
2. **Model Selection** — VGG vs ResNet
3. **Model Training** — 6 emotion classes
4. **Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## 🎯 Key Findings

- **ResNet significantly outperformed VGG** in both accuracy and loss reduction
- VGG showed signs of **underfitting** (~25% training accuracy)
- ResNet achieved **~65% validation accuracy** and better generalization
- Confusion matrix analysis revealed that "fear" and "neutral" are the most challenging categories

| Metric | VGG | ResNet |
|--------|-----|--------|
| Train Accuracy | ~0.25 | > 0.70 |
| Val Accuracy | ~0.40 | ~0.65 |
| Final Loss | > 66 | < 19 |

---

## 🧠 Emotions Classified

- Anger
- Disgust
- Fear
- Happy
- Neutral
- Sad

---

## 📄 Related Paper

A full paper detailing the methodology, experiments, and results is available in this repository (or upon request). It was authored with my teammate **Anrui Wang** as part of our CS210 course.

---

## 🛠️ Dependencies

- Python 3.x
- TensorFlow / Keras
- Librosa
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## ⭐ Acknowledgements

- Datasets: [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D), [TESS](https://tspace.library.utoronto.ca/handle/1807/24487), [RAVDESS](https://zenodo.org/record/1188976)
- Thanks to my teammate Anrui Wang and our professor Dr. Scalzo for guidance on this project

---

*Uploaded with ❤️ as a milestone — the first step into data science.*
