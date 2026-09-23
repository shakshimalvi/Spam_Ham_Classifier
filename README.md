# 📩 Spam & Ham SMS Classifier using Bidirectional LSTM

## 📌 Project Overview

This project is a **Natural Language Processing (NLP) text classification system** that classifies SMS messages into two categories:

* **Ham** — Normal/legitimate message
* **Spam** — Unwanted, promotional, or fraudulent message

A **Bidirectional LSTM (Long Short-Term Memory)** deep learning model is used to understand the sequence and context of words in SMS messages.

The trained model is integrated with a **Streamlit dashboard**, allowing users to enter an SMS and receive a Spam/Ham prediction with its spam probability.

---

## 🎯 Objective

The objective of this project is to develop a deep learning-based NLP model that can automatically classify SMS messages as **Spam or Ham**.

The project covers the complete machine learning workflow, including:

* Exploratory Data Analysis
* Data preprocessing
* Text tokenization
* Sequence padding
* Class imbalance handling
* NLP model development
* Model training
* Performance evaluation
* Model saving
* Streamlit deployment

---

## 📊 Dataset

The dataset contains SMS messages categorized as either **Ham** or **Spam**.

### Dataset Columns

| Column           | Description                         |
| ---------------- | ----------------------------------- |
| `Category`       | Target class: Ham or Spam           |
| `Message`        | SMS message text                    |
| `Message_Length` | Number of characters in the message |

### Class Distribution

| Category  |     Count |
| --------- | --------: |
| Ham       |     4,825 |
| Spam      |       747 |
| **Total** | **5,572** |

The dataset is imbalanced because the number of Ham messages is significantly higher than the number of Spam messages.

---

## 🔎 Exploratory Data Analysis

The following analysis was performed:

* Dataset shape and structure
* Missing-value analysis
* Duplicate-value analysis
* Ham vs Spam distribution
* Message-length analysis
* Data visualization

The EDA helped identify the class imbalance and understand the characteristics of the SMS messages.

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

### 1. Target Encoding

```python
df['Category'] = df['Category'].map({
    'ham': 0,
    'spam': 1
})
```

Where:

* `0` → Ham
* `1` → Spam

### 2. Train-Test Split

The dataset was divided into:

* **80% Training data**
* **20% Testing data**

Stratified splitting was used to maintain the Ham/Spam class distribution.

### 3. Text Tokenization

The SMS messages were converted into numerical sequences using the Keras `Tokenizer`.

```python
Tokenizer(
    num_words=5000,
    oov_token="<OOV>"
)
```

### 4. Sequence Padding

Since SMS messages have different lengths, the sequences were padded to a fixed length of **189 tokens**.

```python
pad_sequences(
    sequences,
    maxlen=189,
    padding='post'
)
```

### 5. Class Imbalance Handling

Class weights were calculated using `compute_class_weight` so that the minority Spam class received greater importance during training.

---

## 🧠 NLP Architecture

The project uses a **Bidirectional LSTM** architecture.

```text
SMS Message
     ↓
Tokenization
     ↓
Sequence Padding
     ↓
Embedding Layer
     ↓
Bidirectional LSTM
     ↓
Dropout
     ↓
Dense Layer
     ↓
Sigmoid
     ↓
Spam / Ham
```

### Model Architecture

```python
model = Sequential([
    Input(shape=(189,)),

    Embedding(
        input_dim=vocab_size,
        output_dim=64
    ),

    Bidirectional(
        LSTM(64)
    ),

    Dropout(0.3),

    Dense(1, activation='sigmoid')
])
```

### Layer Explanation

**Embedding Layer**

Converts token IDs into dense numerical vectors of size 64.

**Bidirectional LSTM**

Processes the SMS sequence in both forward and backward directions to capture contextual information.

**Dropout**

Uses a dropout rate of 0.3 to help reduce overfitting.

**Dense + Sigmoid**

Produces a probability between 0 and 1 for binary classification.

---

## 🏋️ Model Training

The model was trained using:

* **Optimizer:** Adam
* **Loss Function:** Binary Crossentropy
* **Metric:** Accuracy
* **Batch Size:** 32
* **Maximum Epochs:** 10
* **Early Stopping:** Enabled
* **Class Weights:** Enabled

Early stopping was used to stop training when validation performance stopped improving and to restore the best model weights.

---

## 📈 Model Performance

The final Bidirectional LSTM model was evaluated on the unseen test dataset.

### Overall Performance

| Metric            |     Result |
| ----------------- | ---------: |
| Test Accuracy     | **98.39%** |
| Test Loss         | **0.0558** |
| Macro F1-score    |   **0.96** |
| Weighted F1-score |   **0.98** |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
| ----- | --------: | -----: | -------: | ------: |
| Ham   |      0.99 |   1.00 |     0.99 |     966 |
| Spam  |      0.97 |   0.91 |     0.94 |     149 |

### Confusion Matrix

```text
              Predicted
              Ham   Spam

Actual Ham    962     4
Actual Spam    14   135
```

The model correctly classified:

* **962 Ham messages**
* **135 Spam messages**

---

## 🧪 Example Predictions

The trained model was tested with new SMS messages.

| SMS                                                           | Prediction |
| ------------------------------------------------------------- | ---------- |
| Hey, I'll reach home by 7 pm.                                 | Ham        |
| Congratulations! You won a free prize. Click now to claim it! | Spam       |

The application also displays the predicted **Spam probability**.

---

## 💾 Saved Model

The trained model and tokenizer are saved for deployment:

```text
spam_ham_bilstm_model.keras
tokenizer.pkl
```

The tokenizer is saved because the deployment application must use the same word-to-number mapping that was created during training.

---


## 🌐 Live Demo

🚀 **Try the Spam & Ham Classifier:**
**[Live Demo]-- https://spamhamclassifier-blldtfp4bjj3ycyzhyxi5u.streamlit.app/**

Enter any SMS message and the application will predict whether it is **Spam** or **Ham**, along with the Spam probability.

## 🌐 Streamlit Application

The model is integrated into a Streamlit dashboard.

### Application Workflow

```text
User enters SMS
       ↓
Tokenizer
       ↓
Sequence Conversion
       ↓
Padding
       ↓
Saved BiLSTM Model
       ↓
Spam Probability
       ↓
Spam / Ham Result
```

### Run Locally

Clone the repository:

```bash
git clone https://github.com/shakshimalvi/Spam_Ham_Classifier.git
```

Move into the project directory:

```bash
cd Spam_Ham_Classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

---

## 📁 Project Structure

```text
Spam_Ham_Classifier/
│
├── app.py
├── spam_ham_bilstm_model.keras
├── tokenizer.pkl
├── requirements.txt
├── README.md
└── notebook/
    └── Spam_Ham_Classifier.ipynb
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Natural Language Processing (NLP)
* Bidirectional LSTM
* Streamlit
* Git & GitHub

---

## 📚 Key Concepts Demonstrated

* Exploratory Data Analysis
* Text preprocessing
* Tokenization
* Sequence padding
* Word embeddings
* Recurrent Neural Networks
* LSTM
* Bidirectional LSTM
* Binary classification
* Class imbalance
* Class weights
* Early stopping
* Precision
* Recall
* F1-score
* Confusion matrix
* Model serialization
* Streamlit deployment

---

## 🚀 Future Improvements

Possible improvements include:

* Adding a larger and more diverse SMS dataset
* Experimenting with GRU and Transformer-based architectures
* Improving text preprocessing
* Adding prediction history to the dashboard
* Deploying the application on a cloud platform
* Adding model monitoring and feedback collection

---

## 👩‍💻 Author

**Shakshi Malvi**

B.Tech – Computer Science & Engineering

GitHub: [shakshimalvi](https://github.com/shakshimalvi)
