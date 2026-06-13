import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets_code.dataset_lc25000 import LC25000Dataset
from models.backbone import get_efficientnet_b3
from models.student import get_efficientnet_b0_student
from training.kd_trainer import train_one_epoch_kd
from evaluation.utils_eval import evaluate_ce_acc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---- DATA ----
train_ds = LC25000Dataset(split="train")
test_ds = LC25000Dataset(split="test")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

# ---- TEACHER (load trained baseline) ----
teacher = get_efficientnet_b3(num_classes=5).to(device)
teacher.load_state_dict(torch.load("baseline_efficientnet_b3_lc25000.pth", map_location=device))
teacher.eval()
print("Teacher loaded.")

# ---- STUDENT ----
student = get_efficientnet_b0_student(num_classes=5).to(device)
print("Student created (EfficientNet-B0).")

# ---- OPTIM ----
optimizer = optim.Adam(student.parameters(), lr=1e-4)

# ---- KD TRAIN ----
epochs = 5
T = 4.0
alpha = 0.7

for epoch in range(epochs):
    train_loss = train_one_epoch_kd(student, teacher, train_loader, optimizer, device, T=T, alpha=alpha)
    val_loss, val_acc = evaluate_ce_acc(student, test_loader, device)
    print(f"[KD] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

torch.save(student.state_dict(), "student_efficientnet_b0_kd_lc25000.pth")
print("✅ KD student saved: student_efficientnet_b0_kd_lc25000.pth")
