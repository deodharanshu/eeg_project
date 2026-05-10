import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from eegnet_model import EEGNet

def fgsm_attack(model, X, y, epsilon):
    X_adv = X.clone().requires_grad_(True)
    criterion = nn.CrossEntropyLoss()
    outputs = model(X_adv)
    loss = criterion(outputs, y)
    model.zero_grad()
    loss.backward()
    return (X + epsilon * X_adv.grad.sign()).detach()

print("Loading 9-subject dataset...")
dataset = BNCI2014_001()
paradigm = MotorImagery(n_classes=4)
all_X, all_y = [], []
for subject in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
    all_X.append(X)
    all_y.append(y)
X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
_, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

print("Loading 9-subject baseline model...")
model = EEGNet(n_classes=4, n_channels=22, n_timepoints=1001)
model.load_state_dict(torch.load("baseline_model_9subj.pth"))
model.eval()

with torch.no_grad():
    out = model(X_test)
    _, pred = torch.max(out, 1)
    clean_acc = (pred == y_test).float().mean().item()
print(f"Clean accuracy: {clean_acc*100:.2f}%")

print("Running FGSM at all epsilons...")
epsilons = [0.01, 0.05, 0.10, 0.20, 0.30]
attack_accs = []
for eps in epsilons:
    X_adv = fgsm_attack(model, X_test, y_test, eps)
    with torch.no_grad():
        out = model(X_adv)
        _, pred = torch.max(out, 1)
        acc = (pred == y_test).float().mean().item()
    attack_accs.append(acc)
    print(f"  Epsilon {eps:.2f}: {acc*100:.2f}%")

# Generate signal viz at epsilon 0.10
X_adv_demo = fgsm_attack(model, X_test[:5], y_test[:5], 0.10)
trial_idx = 0
channel_idx = 0
clean_signal = X_test[trial_idx, channel_idx].numpy()
adv_signal = X_adv_demo[trial_idx, channel_idx].numpy()

# WIDER figure to fit titles
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].plot(epsilons, [clean_acc*100]*len(epsilons), 'g--', label=f'Baseline (clean): {clean_acc*100:.1f}%', linewidth=2)
axes[0].plot(epsilons, [a*100 for a in attack_accs], 'r-o', label='After FGSM attack', linewidth=2)
axes[0].axhline(y=25, color='gray', linestyle=':', label='Random chance (25%)')
axes[0].set_xlabel('Epsilon (perturbation magnitude)')
axes[0].set_ylabel('Classification Accuracy (%)')
axes[0].set_title('FGSM Attack: Accuracy vs Perturbation', fontsize=12)
axes[0].legend(loc='lower left', fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(clean_signal, 'b-', label='Original EEG', linewidth=1.0, alpha=0.8)
axes[1].plot(adv_signal, 'r-', label='Adversarial EEG', linewidth=1.0, alpha=0.7)
axes[1].set_xlabel('Time points')
axes[1].set_ylabel('Amplitude')
axes[1].set_title('Original vs Adversarial Signal (epsilon=0.10)', fontsize=12)
axes[1].legend(loc='upper right', fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fgsm_attack_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved fgsm_attack_results.png with 9-subject data")