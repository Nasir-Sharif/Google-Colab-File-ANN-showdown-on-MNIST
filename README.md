# Google-Colab-File-ANN-showdown-on-MNIST
# 🤖 Chappie vs Tracy: ANN Showdown on MNIST

Welcome to the ultimate Artificial Neural Network (ANN) battle between **Chappie** and **Tracy**, two hand-coded warriors trained to recognize handwritten digits from the famous **MNIST dataset**. This project was created by **Nasir Sharif** as part of a deep learning exploration using Python, NumPy, and TensorFlow/Keras.

---

## 🧠 Overview

The goal is to build, train, and compare two different ANN architectures from scratch:

- **Chappie**: Lightweight, fast, with a single hidden layer.
- **Tracy**: Deeper, more analytical, with two hidden layers.

Both models are trained on the MNIST dataset and evaluated on a separate `mnist_test.csv` file to determine which performs better.

---

## 📁 Dataset

- **Source**: MNIST (in CSV format)
- **Training Data**: `mnist_train.csv`
- **Testing Data**: `mnist_test.csv`
- **Features**: 784 (28x28 grayscale pixels, normalized)
- **Labels**: Digits from 0 to 9 (one-hot encoded)

---

## ⚙️ Model Architectures

### 🦾 Chappie
- **Input Layer**: 784
- **Hidden Layer**: 64 neurons (ReLU)
- **Output Layer**: 10 neurons (Softmax)
- **Total Layers**: 2 (shallow network)
- **Training Epochs**: 20

### 🤖 Tracy
- **Input Layer**: 784
- **Hidden Layers**: 128 → 64 neurons (ReLU)
- **Output Layer**: 10 neurons (Softmax)
- **Total Layers**: 3 (deeper network)
- **Training Epochs**: 10

---

## 🔍 Preprocessing Steps

- Loaded CSV data using `pandas`
- Normalized pixel values from `0–255` to `0–1`
- One-hot encoded target labels using `to_categorical()`
- Split data using `train_test_split()` from `sklearn`

---

## 📊 Results

After evaluating both models on the unseen test data (`mnist_test.csv`), the final accuracies are:

| Model   | Accuracy on Test Data |
|---------|------------------------|
| **Tracy** | **97.31%** ✅ |
| **Chappie** | **97.25%** ✅ |

➡️ Tracy slightly outperformed Chappie, as expected due to its deeper architecture and greater learning capacity.

---

## 📈 Loss Curve Comparison

A training loss comparison was plotted to visualize learning progress over epochs:

- Chappie: Faster convergence due to simpler model.
- Tracy: Smoother loss curve, capturing more complex patterns.

---

## 🖼️ Visual Predictions

Sample predictions were visualized to assess model performance qualitatively.

```python
def show_sample_predictions(model, X, y_true, count=10):
    predictions = model.predict(X)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_true, axis=1)

    for i in range(count):
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f"Predicted: {pred_labels[i]}, Actual: {true_labels[i]}")
        plt.axis('off')
        plt.show()
```

✅ Technologies Used
Python 3

NumPy

Pandas

TensorFlow (Keras)

Matplotlib

Google Colab


👨‍💻 Author
Nasir Sharif
GitHub: Nasir-Sharif

📌 License
This project is for educational and academic purposes.

🌟 Acknowledgments
Kaggle MNIST CSV Dataset

TensorFlow & Keras for providing high-level neural network APIs.
