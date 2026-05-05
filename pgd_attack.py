import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from eegnet_model import EEGNet

def pgd_attack(model, X, y, epsilon, alpha=None, n_steps=10):
    if alpha is None:
        alpha = epsilon / 4
    
    X_orig = X.clone().detach()
    X_adv = X.clone().detach()
    X_adv = X_adv + torch.empty_like(X_adv).uniform_(-epsilon, epsilon)
    
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(n_steps):
        X_adv = X_adv.clone().detach().requires_grad_(True)
        outputs = model(X_adv)
        loss = criterion(outputs, y)
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()
            delta = torch.clamp(X_adv - X_orig, -epsilon, epsilon)
            X_adv = X_orig + delta
    
    return X_adv.detach()

print("Loading dataset...")
dataset = BNCI2014_001()
paradigm = MotorImagery(n_classes=4)

all_X = []
all_y = []

for subject in [1, 2, 3]:
    X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
    all_X.append(X)
    all_y.append(y)

X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

print("Loading both models...")
baseline_model = EEGNet(n_classes=4, n_channels=22, n_timepoints=1001)
baseline_model.load_state_dict(torch.load("baseline_model.pth"))
baseline_model.eval()

defended_model = EEGNet(n_classes=4, n_channels=22, n_timepoints=1001)
defended_model.load_state_dict(torch.load("adversarial_model.pth"))
defended_model.eval()

epsilons = [0.01, 0.05, 0.1, 0.2, 0.3]
baseline_pgd_accs = []
defended_pgd_accs = []

print("\nRunning PGD attack on baseline model (no defense)...")
for eps in epsilons:
    X_adv = pgd_attack(baseline_model, X_test, y_test, eps, n_steps=10)
    with torch.no_grad():
        outputs = baseline_model(X_adv)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == y_test).float().mean().item()
    baseline_pgd_accs.append(acc)
    print(f"Epsilon: {eps:.2f} | Baseline accuracy under PGD: {acc*100:.2f}%")

print("\nRunning PGD attack on defended model (FGSM adversarial training)...")
for eps in epsilons:
    X_adv = pgd_attack(defended_model, X_test, y_test, eps, n_steps=10)
    with torch.no_grad():
        outputs = defended_model(X_adv)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == y_test).float().mean().item()
    defended_pgd_accs.append(acc)
    print(f"Epsilon: {eps:.2f} | Defended accuracy under PGD: {acc*100:.2f}%")

fgsm_baseline = [0.5202, 0.3526, 0.1676, 0.0202, 0.0000]
fgsm_defended = [0.5636, 0.4798, 0.3786, 0.2052, 0.1040]

print("\nGenerating comparison plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epsilons, [acc*100 for acc in fgsm_baseline], 'r--o', label='FGSM attack', linewidth=2)
axes[0].plot(epsilons, [acc*100 for acc in baseline_pgd_accs], 'r-s', label='PGD attack', linewidth=2)
axes[0].axhline(y=25, color='gray', linestyle=':', label='Random chance (25%)')
axes[0].set_xlabel('Epsilon (perturbation magnitude)')
axes[0].set_ylabel('Classification Accuracy (%)')
axes[0].set_title('Baseline Model: FGSM vs PGD')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epsilons, [acc*100 for acc in fgsm_defended], 'b--o', label='FGSM attack', linewidth=2)
axes[1].plot(epsilons, [acc*100 for acc in defended_pgd_accs], 'b-s', label='PGD attack', linewidth=2)
axes[1].axhline(y=25, color='gray', linestyle=':', label='Random chance (25%)')
axes[1].set_xlabel('Epsilon (perturbation magnitude)')
axes[1].set_ylabel('Classification Accuracy (%)')
axes[1].set_title('Defended Model: FGSM vs PGD')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('pgd_comparison.png', dpi=150)
plt.show()
print("Plot saved to pgd_comparison.png")

print("\nSummary:")
print(f"{'Epsilon':<10}{'Baseline FGSM':<16}{'Baseline PGD':<16}{'Defended FGSM':<16}{'Defended PGD':<16}")
for i, eps in enumerate(epsilons):
    print(f"{eps:<10.2f}{fgsm_baseline[i]*100:<16.2f}{baseline_pgd_accs[i]*100:<16.2f}{fgsm_defended[i]*100:<16.2f}{defended_pgd_accs[i]*100:<16.2f}")