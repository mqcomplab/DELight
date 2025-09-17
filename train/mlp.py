import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Load your data
X_train = np.load('fps_undersampled.npy', mmap_mode='r')
X_test = np.load('../data/fps_test.npy', mmap_mode='r')
y_train = np.load('y_train.npy', mmap_mode='r')
y_test = np.load('y_test.npy', mmap_mode='r')

# 2. Scale input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Build binary classification model
model = Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary output
])

# 4. Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. Train
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# 6. Predict
y_scores = model.predict(X_test).flatten()       # Probabilities
y_pred = (y_scores >= 0.5).astype(int)            # Class labels (threshold at 0.5)

# 7. Metrics
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_scores)

# 8. Report
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {auc:.4f}")

with open('mlp_log.csv', 'w') as f:
    f.write("Model,Accuracy,Precision,Recall,F1,ROC_AUC\n")
    f.write(f"MLP,{acc:.4f},{precision:.4f},{recall:.4f},{f1:.4f},{auc:.4f}\n")