import torch
import torch.nn.functional as F
from tqdm import tqdm


def kd_loss_fn(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    """
    alpha: weight for KD (soft targets)
    (1-alpha): weight for CE (hard labels)
    T: temperature
    """
    ce = F.cross_entropy(student_logits, labels)

    # KL divergence between softened probabilities
    p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    kl = F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)

    return alpha * kl + (1 - alpha) * ce


def train_one_epoch_kd(student, teacher, loader, optimizer, device, T=4.0, alpha=0.7):
    student.train()
    teacher.eval()

    total_loss = 0.0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits = teacher(images)

        student_logits = student(images)
        loss = kd_loss_fn(student_logits, teacher_logits, labels, T=T, alpha=alpha)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
