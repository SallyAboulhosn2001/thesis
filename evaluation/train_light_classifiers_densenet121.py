import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

data = torch.load("nct_features_densenet121.pt")
X = data["X"].numpy()
y = data["y"].numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

logreg = LogisticRegression(max_iter=2000, n_jobs=-1)
logreg.fit(X_train, y_train)
y_lr = logreg.predict(X_test)

print(
    f"[DenseNet121 + LR] Accuracy: {accuracy_score(y_test, y_lr):.4f} | "
    f"Macro-F1: {f1_score(y_test, y_lr, average='macro'):.4f}"
)

svm = LinearSVC(max_iter=3000)
svm.fit(X_train, y_train)
y_svm = svm.predict(X_test)

print(
    f"[DenseNet121 + SVM] Accuracy: {accuracy_score(y_test, y_svm):.4f} | "
    f"Macro-F1: {f1_score(y_test, y_svm, average='macro'):.4f}"
)
