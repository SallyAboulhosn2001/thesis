import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Load fused features
data = torch.load("nct_features_fused_b0kd_densenet.pt")
X = data["X"].numpy()
y = data["y"].numpy()

print("Original feature shape:", X.shape)

# Apply PCA
pca = PCA(n_components=500)
X_reduced = pca.fit_transform(X)

print("Reduced feature shape:", X_reduced.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression
lr = LogisticRegression(max_iter=3000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("[FUSED + PCA + LR] Accuracy:",
      accuracy_score(y_test, y_pred_lr))
print("[FUSED + PCA + LR] Macro-F1:",
      f1_score(y_test, y_pred_lr, average="macro"))

# SVM
svm = LinearSVC(max_iter=5000)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("[FUSED + PCA + SVM] Accuracy:",
      accuracy_score(y_test, y_pred_svm))
print("[FUSED + PCA + SVM] Macro-F1:",
      f1_score(y_test, y_pred_svm, average="macro"))
