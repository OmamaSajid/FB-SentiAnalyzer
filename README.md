# 🧠 FB-SentiAnalyzer - Fcaebook Sentiment Classifier

**FB-SentiAnalyzer** is a machine learning project that classifies the sentiment of facebook comments as **Positive (1)** or **Neutral (0)** or **Negative (2)**  using the **fb_sentiment.csv** dataset. It uses natural language processing (NLP) and logistic regression to train a model that can analyze comments tone and predict sentiment in real time.

---

## 🔍 Project Overview

This project:
- Loads and preprocesses tweets using NLTK.
- Converts text into TF-IDF vectors.
- Trains a logistic regression model for binary sentiment classification.
- Predicts sentiment of new, user-written tweets directly from code input.

---

## 📊 Dataset: fb_sentiment.csv

- **Source**: [Kaggle - fb_sentiment](https://www.kaggle.com/code/mortena/facebook-comments-sentiment-analysis/input)
- **Original Labels**:
  - `N` → Negative
  - `P` → Positive
  - `O` → Neutral

> 🛠 **Modification**: For simplicity and binary classification, this project maps label, so the sentiment labels are now:
>
> - `N` → 2
  - `P` → 1
  - `O` → 0

- **Fields Used**:
  - `Label`: sentiment label (0 or 1)
  - `fb_post`: the facebook post content

---

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- NLTK (Stopwords, Stemming)
- scikit-learn (TF-IDF, Logistic Regression)
- Regex (`re`)

---
---

## 🛠️ How to Use This Project

You have **two options** to run TweetSensei:

### ✅ 1. Use Pre-trained Model (Quick Start)


- Upload these files to your Colab session:
  - `trained_fb_model.sav`
 

- Then, load them like this:
  ```python
  import pickle

  model = pickle.load(open('trained_model.sav', 'rb'))
If you want to train the model from the beginning:

Open the notebook tweetsensei.ipynb in Google Colab.

Download the dataset from Kaggle:

kaggle datasets download -d kazanova/fb-sentiment
unzip sentiment140.zip

And enjoyyy


Thanks for checking out FB-SentiAnalyzer !
🐦 Happy Tweet Sniffing — and see you next project!
Bye Bye! 👋


