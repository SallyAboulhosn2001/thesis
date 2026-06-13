import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib

# load features
data = torch.load("lc25000_features_densenet121.pt")  # or LC25000 features
X = data["X"]
y = data["y"]

print("Features:", X.shape)

# -------- SVM --------
svm = SVC(kernel="rbf", probability=True)
svm.fit(X, y)
joblib.dump(svm, "svm_literature.pkl")

# -------- LR --------
lr = LogisticRegression(max_iter=3000)
lr.fit(X, y)
joblib.dump(lr, "lr_literature.pkl")

print("✅ Literature hybrids trained")