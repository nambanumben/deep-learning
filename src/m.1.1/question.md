---

### **QUESTION 1: SUPERVISED LEARNING**  
Import the Iris dataset from `sklearn` library and keep 75% for the training set  

1. Use logistic regression, SVM, Random forest, and decision Tree and get the score for each Model **(10 marks)**  
2. Use the `cross_val_score` object with 5 slots, and Measure the different scores for logistic regression, SVM, Random forest, and decision Tree. **(10 marks)**  
3. Plot the confusion Matrix for the model with the best accuracy **(10 marks)**  

---

### **QUESTION 2: SUPERVISED LEARNING**  
Import the Digit dataset from `sklearn` library  

1. Using the K-Means, vary the number of Cluster form 1 to 20, plot the K against the SSE (Sum of Squared Errors), and deduce the optimal value of K **(10 marks)**  

---

### **QUESTION 3: DEEP LEARNING**  
Import the `Cifar10` dataset from `tensorflow.keras` library. Build a Convolutional Neural Network, made up of 03 stacks (Convolution, Relu, Pooling), coupled to a fully connected output layer,  

1. Compile, and train it **(10 marks)**  
2. Generate the summary of the network **(10 marks)**  
3. Plot the confusion Matrix for the model **(10 marks)**  

---

## **Answers to All Questions**

I'll provide Python solutions for each question.

---

### **QUESTION 1: SUPERVISED LEARNING**
#### **Step 1: Import Libraries and Load the Dataset**
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=42)
```

#### **Step 2: Train and Evaluate Models**
```python
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores[name] = accuracy_score(y_test, y_pred)

best_model = max(scores, key=scores.get)
print("Model Accuracy Scores:", scores)
print(f"Best Model: {best_model}")
```

#### **Step 3: Cross-Validation**
```python
for name, model in models.items():
    cv_scores = cross_val_score(model, iris.data, iris.target, cv=5)
    print(f"{name} Cross-validation scores: {cv_scores.mean():.4f}")
```

#### **Step 4: Plot Confusion Matrix for Best Model**
```python
best_clf = models[best_model]
y_pred_best = best_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix for {best_model}")
plt.show()
```

---

### **QUESTION 2: SUPERVISED LEARNING (K-Means Clustering on Digits Dataset)**
```python
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import numpy as np

# Load dataset
digits = load_digits()
X = digits.data

# Compute Sum of Squared Errors (SSE) for K from 1 to 20
sse = []
k_values = range(1, 21)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# Plot SSE vs. K
plt.figure(figsize=(8, 5))
plt.plot(k_values, sse, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method to Find Optimal K')
plt.show()

# Optimal K is typically where the SSE curve bends
```

---

### **QUESTION 3: DEEP LEARNING (CNN on CIFAR-10)**
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize images
X_train, X_test = X_train / 255.0, X_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
```

#### **Step 2: Generate Model Summary**
```python
model.summary()
```

#### **Step 3: Plot Confusion Matrix**
```python
import numpy as np
from sklearn.metrics import confusion_matrix

# Predict labels
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for CIFAR-10 CNN Model')
plt.show()
```
