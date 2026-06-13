import torch

# Load feature files
b0 = torch.load("nct_features_student_b0_kd.pt")
dn = torch.load("nct_features_densenet121.pt")

# Correct keys
X1, y1 = b0["X"], b0["y"]
X2, y2 = dn["X"], dn["y"]

# Make sure labels match
assert torch.equal(y1, y2), "Labels do not match!"

# Concatenate features
X_fused = torch.cat([X1, X2], dim=1)

# Save fused features
torch.save({
    "X": X_fused,
    "y": y1
}, "nct_features_fused_b0kd_densenet.pt")

print("✅ Fusion complete.")
print("Fused shape:", X_fused.shape)
