import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from eegnet_model import EEGNet
from shallow_cnn import ShallowCNN

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

print("Loading all three models...")
surrogate = ShallowCNN(n_classes=4, n_channels=22, n_timepoints=1001)
surrogate.load_state_dict(torch.load("shallow_model.pth"))
surrogate.eval()

baseline = EEGNet(n_classes=4, n_channels=22, n_timepoints=1001)
baseline.load_state_dict(torch.load("baseline_model.pth"))
baseline.eval()

defended = EEGNet(n_classes=4, n_channels=22, n_timepoints=1001)
defended.load_state_dict(torch.load("adversarial_model.pth"))
defended.eval()

with torch.no_grad():
    out = surrogate(X_test)
    _, pred = torch.max(out, 1)
    surr_clean = (pred == y_test).float().mean().item()
    
    out = baseline(X_test)
    _, pred = torch.max(out, 1)
    base_clean = (pred == y_test).float().mean().item()
    
    out = defended(X_test)
    _, pred = torch.max(out, 1)
    def_clean = (pred == y_test).float().mean().item()

print(f"\nClean accuracies:")
print(f"Surrogate (ShallowCNN): {surr_clean*100:.2f}%")
print(f"Baseline EEGNet: {base_clean*100:.2f}%")
print(f"Defended EEGNet: {def_clean*100:.2f}%")

epsilons = [0.01, 0.05, 0.1, 0.2, 0.3]

surrogate_self_accs = []
baseline_transfer_accs = []
defended_transfer_accs = []

print("\nGenerating adversarial examples on surrogate, evaluating transfer to EEGNets...")
for eps in epsilons:
    X_adv = fgsm_attack(surrogate, X_test, y_test, eps)
    
    with torch.no_grad():
        out = surrogate(X_adv)
        _, pred = torch.max(out, 1)
        surr_acc = (pred == y_test).float().mean().item()
    surrogate_self_accs.append(surr_acc)
    
    with torch.no_grad():
        out = baseline(X_adv)
        _, pred = torch.max(out, 1)
        base_acc = (pred == y_test).float().mean().item()
    baseline_transfer_accs.append(base_acc)
    
    with torch.no_grad():
        out = defended(X_adv)
        _, pred = torch.max(out, 1)
        def_acc = (pred == y_test).float().mean().item()
    defended_transfer_accs.append(def_acc)
    
    print(f"Epsilon: {eps:.2f} | Surrogate: {surr_acc*100:.2f}% | Baseline EEGNet: {base_acc*100:.2f}% | Defended EEGNet: {def_acc*100:.2f}%")

whitebox_baseline = [0.5202, 0.3526, 0.1676, 0.0202, 0.0000]
whitebox_defended = [0.5636, 0.4798, 0.3786, 0.2052, 0.1040]

print("\nGenerating comparison plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epsilons, [acc*100 for acc in whitebox_baseline], 'r-o', label='White-box (FGSM on EEGNet)', linewidth=2)
axes[0].plot(epsilons, [acc*100 for acc in baseline_transfer_accs], 'r--s', label='Black-box (transfer from ShallowCNN)', linewidth=2)
axes[0].axhline(y=base_clean*100, color='g', linestyle=':', label=f'Clean ({base_clean*100:.1f}%)')
axes[0].axhline(y=25, color='gray', linestyle=':', label='Random chance (25%)')
axes[0].set_xlabel('Epsilon (perturbation magnitude)')
axes[0].set_ylabel('Classification Accuracy (%)')
axes[0].set_title('Baseline EEGNet: White-box vs Black-box Attack')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epsilons, [acc*100 for acc in whitebox_defended], 'b-o', label='White-box (FGSM on EEGNet)', linewidth=2)
axes[1].plot(epsilons, [acc*100 for acc in defended_transfer_accs], 'b--s', label='Black-box (transfer from ShallowCNN)', linewidth=2)
axes[1].axhline(y=def_clean*100, color='g', linestyle=':', label=f'Clean ({def_clean*100:.1f}%)')
axes[1].axhline(y=25, color='gray', linestyle=':', label='Random chance (25%)')
axes[1].set_xlabel('Epsilon (perturbation magnitude)')
axes[1].set_ylabel('Classification Accuracy (%)')
axes[1].set_title('Defended EEGNet: White-box vs Black-box Attack')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('transfer_attack_results.png', dpi=150)
plt.show()
print("Plot saved to transfer_attack_results.png")

print("\nSummary table:")
print(f"{'Epsilon':<10}{'Surr self':<12}{'Base WB':<12}{'Base BB':<12}{'Def WB':<12}{'Def BB':<12}")
for i, eps in enumerate(epsilons):
    print(f"{eps:<10.2f}{surrogate_self_accs[i]*100:<12.2f}{whitebox_baseline[i]*100:<12.2f}{baseline_transfer_accs[i]*100:<12.2f}{whitebox_defended[i]*100:<12.2f}{defended_transfer_accs[i]*100:<12.2f}")