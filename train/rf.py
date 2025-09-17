from sklearn.ensemble import RandomForestClassifier
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

    print(f"Random Forest Accuracy: {acc:.4f}")
    print(f"Random Forest Precision: {precision:.4f}")
    print(f"Random Forest Recall: {recall:.4f}")
    print(f"Random Forest F1-score: {f1:.4f}")
    print(f"Random Forest ROC AUC: {auc:.4f}")

    return acc, precision, recall, f1, auc, y_pred

# Train and evaluate
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
metrics = evaluate_model(rf_model, X_test, y_test)
np.save('rf_pred.npy', metrics[-1])
joblib.dump(rf_model, 'rf_model.pkl')

with open('random_forest_log.csv', 'w') as f:
    f.write("Model,Accuracy,Precision,Recall,F1,ROC_AUC\n")
    f.write(f"Random Forest,{','.join(f'{m:.4f}' for m in metrics[:-1])}\n")
