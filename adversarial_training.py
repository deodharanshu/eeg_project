import torch
import torch.nn as nn
import torch.optim as optim
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
    return (X + perturbation).detach()

print("Loading dataset...")
dataset = BNCI2014_001()
paradigm = MotorImagery(n_classes=4)

all_X, all_y = [], []
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

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

model = EEGNet(n_classes=4, n_channels=22, n_timepoints=1001)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epsilon_train = 0.1
n_epochs = 50
batch_size = 32

print("Running adversarial training...")
for epoch in range(n_epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0

    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_X = X_train[indices]
        batch_y = y_train[indices]

        X_adv = fgsm_attack(model, batch_X, batch_y, epsilon_train)
        combined_X = torch.cat([batch_X, X_adv], dim=0)
        combined_y = torch.cat([batch_y, batch_y], dim=0)

        model.train()
        optimizer.zero_grad()
        outputs = model(combined_X)
        loss = criterion(outputs, combined_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs, 1)
            clean_acc = (predicted == y_test).float().mean().item()
        print(f"Epoch {epoch+1}/50 | Loss: {epoch_loss:.4f} | Clean Accuracy: {clean_acc:.4f}")

print("\nEvaluating defense...")
model.eval()

with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    clean_acc = (predicted == y_test).float().mean().item()
print(f"Clean accuracy after adversarial training: {clean_acc*100:.2f}%")

epsilons = [0.01, 0.05, 0.1, 0.2, 0.3]
baseline_accs = [52.02, 35.26, 16.76, 2.02, 0.00]
defended_accs = []

for eps in epsilons:
    X_adv = fgsm_attack(model, X_test, y_test, eps)
    with torch.no_grad():
        outputs = model(X_adv)
        _, predicted = torch.max(outputs, 1)
        adv_acc = (predicted == y_test).float().mean().item()
    defended_accs.append(adv_acc * 100)
    print(f"Epsilon: {eps:.2f} | Accuracy under attack with defense: {adv_acc*100:.2f}%")

torch.save(model.state_dict(), "adversarial_model.pth")
print("\nDefended model saved to adversarial_model.pth")

print("\nGenerating comparison plot...")
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(epsilons, [clean_acc*100]*len(epsilons), 'g--', label=f'Clean baseline ({clean_acc*100:.1f}%)', linewidth=2)
ax.plot(epsilons, baseline_accs, 'r-o', label='No defense (baseline model)', linewidth=2)
ax.plot(epsilons, defended_accs, 'b-s', label='With adversarial training defense', linewidth=2)
ax.axhline(y=25, color='gray', linestyle=':', label='Random chance (25%)', linewidth=1.5)

ax.set_xlabel('Epsilon (perturbation magnitude)')
ax.set_ylabel('Classification Accuracy (%)')
ax.set_title('FGSM Attack: Baseline vs Adversarial Training Defense')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('defense_comparison.png', dpi=150)
plt.show()
print("Plot saved to defense_comparison.png")