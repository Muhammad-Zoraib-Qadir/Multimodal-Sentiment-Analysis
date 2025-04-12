# 📸 Multimodal Sentiment Analyzer  
**Predict image sentiment using text + visual analysis**  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-green)](https://opencv.org/)

A hybrid AI system that predicts **positive**/**negative**/**neutral** labels for images by analyzing:
1. Embedded text (OCR + NLP)
2. Visual features (OpenCV)
3. Existing metadata/labels

![Demo Pipeline](https://via.placeholder.com/800x400.png/009688/ffffff?text=Image+→+Text+OCR+→+Sentiment+Analysis+→+Prediction)

## 🌟 Features
- **Multi-source Analysis**  
  🔍 Text extraction from images using OCR  
  🎨 Visual feature extraction with OpenCV/skimage  
  📊 Metadata integration with Pandas  

- **Advanced Processing**  
  🧹 NLP text cleaning (stopwords, regex)  
  📸 Image preprocessing (normalization, resizing)  

- **Model Evaluation**  
  📈 Accuracy/F1-score metrics  
  🤖 Confusion matrix visualization  

## 🛠 Tech Stack
**Core Libraries**
    ```python
      # Computer Vision
      import cv2
      from skimage.io import imread
      
      # Data Handling
      import numpy as np
      import pandas as pd
      
      # ML Pipeline
      from sklearn.model_selection import train_test_split
      from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
      
      # NLP
      import nltk
      from nltk.corpus import stopwords
      import re
      
      # Utilities
      import matplotlib.pyplot as plt
      import os

## 📜 License 
[![MIT License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)  
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Show Your Support**  
[![GitHub Stars](https://img.shields.io/github/stars/yourusername/repo-name?style=social)](https://github.com/yourusername/repo-name/stargazers)  
🌟 **Star this repo** if you found it useful! 

**Found an Issue?**  
[![Open Issues](https://img.shields.io/github/issues/yourusername/repo-name)](https://github.com/yourusername/repo-name/issues)  
🐞 **Report bugs/improvements** in [GitHub Issues](https://github.com/yourusername/repo-name/issues)
</div>
  
