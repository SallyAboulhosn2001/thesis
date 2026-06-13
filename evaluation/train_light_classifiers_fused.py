import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Load fused features
data = torch.load("nct_features_fused_b0kd_densenet.pt")
X = data["X"].numpy()
y = data["y"].numpy()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Logistic Regression ----
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("[FUSED + LR] Accuracy:", accuracy_score(y_test, y_pred_lr))
print("[FUSED + LR] Macro-F1:", f1_score(y_test, y_pred_lr, average="macro"))

# ---- Linear SVM ----
svm = LinearSVC(max_iter=5000)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("[FUSED + SVM] Accuracy:", accuracy_score(y_test, y_pred_svm))
print("[FUSED + SVM] Macro-F1:", f1_score(y_test, y_pred_svm, average="macro"))
