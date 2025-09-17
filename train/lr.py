from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import numpy as np
import joblib

# Load data
X_train = np.load('fps_undersampled.npy', mmap_mode='r')
X_test = np.load('../data/fps_test.npy', mmap_mode='r')
y_train = np.load('y_train.npy', mmap_mode='r')
y_test = np.load('y_test.npy', mmap_mode='r')

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_scores)

    print(f"Logistic Regression Accuracy: {acc:.4f}")
    print(f"Logistic Regression Precision: {precision:.4f}")
    print(f"Logistic Regression Recall: {recall:.4f}")
    print(f"Logistic Regression F1-score: {f1:.4f}")
    print(f"Logistic Regression ROC AUC: {auc:.4f}")

    return acc, precision, recall, f1, auc, y_pred

# Train and evaluate
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
metrics = evaluate_model(lr_model, X_test, y_test)
np.save('lr_pred.npy', metrics[-1])
joblib.dump(lr_model, 'lr_model.pkl')

with open('logistic_regression_log.csv', 'w') as f:
    f.write("Model,Accuracy,Precision,Recall,F1,ROC_AUC\n")
    f.write(f"Logistic Regression,{','.join(f'{m:.4f}' for m in metrics[:-1])}\n")
