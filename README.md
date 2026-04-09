# Horizontal Ensemble from Partial Dataset

A majority-voting neural network ensemble trained on a loan approval dataset. Five independent neural networks are trained with checkpoint saving, and their predictions are combined via majority vote.

---

## What it does

- Trains 5 identical feedforward neural networks independently on the same dataset
- Each model saves only its best checkpoint (monitored by `val_accuracy`)
- Predictions from all 5 best models are collected and combined via majority vote
- Final ensemble accuracy is printed on the held-out test set

---

## Data preprocessing pipeline

```
Raw CSV
→ SimpleImputer (most_frequent) — fills missing values
→ LabelEncoder — encodes categorical columns
→ MinMaxScaler — normalizes numerical features
→ train_test_split (80/20)
```

---

## Model Architecture (per network)

```
Dense(64, relu)
→ Dense(32, relu)
→ Dense(1, sigmoid)
```

Compiled with `binary_crossentropy` + `adam`. Trained for 50 epochs, batch size 32, 20% validation split.

---

## Ensemble decision

```python
# Majority vote across 5 models
ensemble_predictions = (sum(predictions) >= (len(predictions) / 2)).astype(int)
```

---

## Tech Stack

`Python` · `TensorFlow / Keras` · `scikit-learn` · `Pandas` · `NumPy`

---

## Usage

```bash
pip install tensorflow scikit-learn pandas numpy
# Place loan_data.csv in the working directory
python horizontal_ensemble.py
```

---

## Notes

University coursework project. Dataset: loan approval tabular data. The ensemble approach is designed to reduce overfitting variance by averaging out individual model errors through democratic voting.
