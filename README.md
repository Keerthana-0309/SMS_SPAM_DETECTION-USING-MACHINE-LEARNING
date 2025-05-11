# 📱 SMS Spam Detection using Machine Learning

A machine learning project to detect spam messages in SMS texts using NLP techniques and classification algorithms.

---

## 🔍 Overview

This project focuses on detecting spam in SMS messages using various machine learning models. By applying Natural Language Processing (NLP) techniques, the system can classify messages as either **spam** or **ham** (not spam) with high accuracy.

---

## 📦 Features

* Preprocessing of SMS text (tokenization, stopword removal, stemming)
* Feature extraction using Bag of Words / TF-IDF
* Training multiple machine learning models

  * Naive Bayes
  * Support Vector Machine (SVM)
  * Logistic Regression
  * Random Forest
* Model evaluation and comparison
* GUI/Web interface for real-time message prediction (optional)

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries:** Scikit-learn, NLTK, Pandas, NumPy, Matplotlib, Seaborn
* **Dataset:** [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
* **IDE/Environment:** Jupyter Notebook / VS Code / Colab

---



---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/sms-spam-detection.git
cd sms-spam-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook

```bash
jupyter notebook notebooks/spam_detection.ipynb
```

---

## 📊 Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Naive Bayes         | 98.2%    |
| Logistic Regression | 97.5%    |
| SVM                 | 97.9%    |
| Random Forest       | 96.8%    |

---

## 📌 Future Improvements

* Add deep learning models (e.g., LSTM)
* Deploy as a web app using Flask or Streamlit
* Create API for real-time classification
* Add support for multiple languages

---
## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or new features.

---

