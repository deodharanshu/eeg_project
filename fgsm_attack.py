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
    perturbation = epsilon * X_adv.grad.sign()
    X_adv = X + perturbation
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

print("Loading trained model...")
model = EEGNet(n_classes=4, n_channels=22, n_timepoints=1001)
model.load_state_dict(torch.load("baseline_model.pth"))
model.eval()

with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    baseline_acc = (predicted == y_test).float().mean().item()
print(f"Baseline accuracy (clean data): {baseline_acc*100:.2f}%")

epsilons = [0.01, 0.05, 0.1, 0.2, 0.3]
attacked_accs = []

print("\nRunning FGSM attack at different epsilon values...")
for eps in epsilons:
    X_adv = fgsm_attack(model, X_test, y_test, eps)
    with torch.no_grad():
        outputs = model(X_adv)
        _, predicted = torch.max(outputs, 1)
        adv_acc = (predicted == y_test).float().mean().item()
    attacked_accs.append(adv_acc)
    print(f"Epsilon: {eps:.2f} | Accuracy after attack: {adv_acc*100:.2f}%")

print("\nGenerating visualizations...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epsilons, [baseline_acc*100]*len(epsilons), 'g--', label='Baseline (clean)', linewidth=2)
axes[0].plot(epsilons, [acc*100 for acc in attacked_accs], 'r-o', label='After FGSM attack', linewidth=2)
axes[0].axhline(y=25, color='gray', linestyle=':', label='Random chance (25%)')
axes[0].set_xlabel('Epsilon (perturbation magnitude)')
axes[0].set_ylabel('Classification Accuracy (%)')
axes[0].set_title('FGSM Attack: Accuracy vs Perturbation Magnitude')
axes[0].legend()
axes[0].grid(True)

sample_idx = 0
eps = 0.1
X_sample = X_test[sample_idx:sample_idx+1]
y_sample = y_test[sample_idx:sample_idx+1]
X_adv_sample = fgsm_attack(model, X_sample, y_sample, eps)

channel = 0
time_points = np.arange(1001)
axes[1].plot(time_points, X_sample[0, channel].numpy(), 'b-', label='Original EEG', alpha=0.8)
axes[1].plot(time_points, X_adv_sample[0, channel].numpy(), 'r-', label='Adversarial EEG', alpha=0.8)
axes[1].set_xlabel('Time points')
axes[1].set_ylabel('Amplitude')
axes[1].set_title(f'Original vs Adversarial EEG Signal (Channel 1, epsilon={eps})')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('fgsm_attack_results.png', dpi=150)
plt.show()
print("Plot saved to fgsm_attack_results.png")

print("\nSummary:")
print(f"Baseline accuracy: {baseline_acc*100:.2f}%")
print(f"Accuracy at epsilon=0.1: {attacked_accs[2]*100:.2f}%")
print(f"Accuracy at epsilon=0.3: {attacked_accs[4]*100:.2f}%")
print(f"Random chance: 25.00%")