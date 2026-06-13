import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ---- LOAD FEATURES ----
data = torch.load("nct_features_student_b0_kd.pt")
X = data["X"].numpy()
y = data["y"].numpy()

# ---- SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- LOGISTIC REGRESSION ----
logreg = LogisticRegression(max_iter=2000, n_jobs=-1)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average="macro")

print(f"[STUDENT + LR] Accuracy: {acc_lr:.4f} | Macro-F1: {f1_lr:.4f}")

# ---- LINEAR SVM ----
svm = LinearSVC(max_iter=3000)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

acc_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average="macro")

print(f"[STUDENT + SVM] Accuracy: {acc_svm:.4f} | Macro-F1: {f1_svm:.4f}")
