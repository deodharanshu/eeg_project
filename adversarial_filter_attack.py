import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from eegnet_model import EEGNet

class AdversarialFilter(nn.Module):
    def __init__(self, filter_length=20, n_channels=22):
        super(AdversarialFilter, self).__init__()
        self.filter_length = filter_length
        self.n_channels = n_channels
        self.filter_coeffs = nn.Parameter(
            torch.randn(n_channels, 1, filter_length) * 0.1
        )
    
    def forward(self, x, epsilon):
        padding = self.filter_length // 2
        perturbation = F.conv1d(
            x, 
            self.filter_coeffs, 
            padding=padding, 
            groups=self.n_channels
        )
        if perturbation.size(2) != x.size(2):
            perturbation = perturbation[:, :, :x.size(2)]
        
        signal_std = x.std()
        budget = epsilon * signal_std
        
        pert_std = perturbation.std()
        if pert_std > 0:
            perturbation = perturbation * (budget / pert_std)
        
        perturbation = torch.clamp(perturbation, -3 * budget, 3 * budget)
        
        return x + perturbation

def train_adversarial_filter(model, X_train, y_train, epsilon, n_epochs=50, batch_size=32, filter_length=20):
    adv_filter = AdversarialFilter(filter_length=filter_length, n_channels=22)
    optimizer = optim.Adam(adv_filter.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"  Training filter at epsilon={epsilon:.2f}...")
    for epoch in range(n_epochs):
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0
        
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X = X_train[indices]
            batch_y = y_train[indices]
            
            X_adv = adv_filter(batch_X, epsilon)
            outputs = model(X_adv)
            
            loss = -criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    
    for param in model.parameters():
        param.requires_grad = True
    
    return adv_filter

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

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Signal std: {X_train.std().item():.4f}")

print("\nLoading both models...")
baseline = EEGNet(n_classes=4, n_channels=22, n_timepoints=1001)
baseline.load_state_dict(torch.load("baseline_model.pth"))
baseline.eval()

defended = EEGNet(n_classes=4, n_channels=22, n_timepoints=1001)
defended.load_state_dict(torch.load("adversarial_model.pth"))
defended.eval()

with torch.no_grad():
    out = baseline(X_test)
    _, pred = torch.max(out, 1)
    base_clean = (pred == y_test).float().mean().item()
    
    out = defended(X_test)
    _, pred = torch.max(out, 1)
    def_clean = (pred == y_test).float().mean().item()

print(f"Baseline clean: {base_clean*100:.2f}%")
print(f"Defended clean: {def_clean*100:.2f}%")

epsilons = [0.01, 0.05, 0.1, 0.2, 0.3]
baseline_filter_accs = []
defended_filter_accs = []

print("\n=== Training universal adversarial filter on baseline model ===")
for eps in epsilons:
    adv_filter = train_adversarial_filter(baseline, X_train, y_train, eps)
    
    adv_filter.eval()
    with torch.no_grad():
        X_adv = adv_filter(X_test, eps)
        out = baseline(X_adv)
        _, pred = torch.max(out, 1)
        acc = (pred == y_test).float().mean().item()
    baseline_filter_accs.append(acc)
    print(f"  Epsilon: {eps:.2f} | Baseline accuracy under filter attack: {acc*100:.2f}%")

print("\n=== Training universal adversarial filter on defended model ===")
for eps in epsilons:
    adv_filter = train_adversarial_filter(defended, X_train, y_train, eps)
    
    adv_filter.eval()
    with torch.no_grad():
        X_adv = adv_filter(X_test, eps)
        out = defended(X_adv)
        _, pred = torch.max(out, 1)
        acc = (pred == y_test).float().mean().item()
    defended_filter_accs.append(acc)
    print(f"  Epsilon: {eps:.2f} | Defended accuracy under filter attack: {acc*100:.2f}%")

fgsm_baseline = [0.5202, 0.3526, 0.1676, 0.0202, 0.0000]
fgsm_defended = [0.5636, 0.4798, 0.3786, 0.2052, 0.1040]

print("\nGenerating comparison plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epsilons, [acc*100 for acc in fgsm_baseline], 'r-o', label='FGSM (per-trial)', linewidth=2)
axes[0].plot(epsilons, [acc*100 for acc in baseline_filter_accs], 'm-s', label='Universal filter', linewidth=2)
axes[0].axhline(y=base_clean*100, color='g', linestyle=':', label=f'Clean ({base_clean*100:.1f}%)')
axes[0].axhline(y=25, color='gray', linestyle=':', label='Random chance (25%)')
axes[0].set_xlabel('Epsilon (relative to signal std)')
axes[0].set_ylabel('Classification Accuracy (%)')
axes[0].set_title('Baseline EEGNet: FGSM vs Universal Filter Attack')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epsilons, [acc*100 for acc in fgsm_defended], 'b-o', label='FGSM (per-trial)', linewidth=2)
axes[1].plot(epsilons, [acc*100 for acc in defended_filter_accs], 'm-s', label='Universal filter', linewidth=2)
axes[1].axhline(y=def_clean*100, color='g', linestyle=':', label=f'Clean ({def_clean*100:.1f}%)')
axes[1].axhline(y=25, color='gray', linestyle=':', label='Random chance (25%)')
axes[1].set_xlabel('Epsilon (relative to signal std)')
axes[1].set_ylabel('Classification Accuracy (%)')
axes[1].set_title('Defended EEGNet: FGSM vs Universal Filter Attack')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('filter_attack_results.png', dpi=150)
plt.show()
print("Plot saved to filter_attack_results.png")

print("\nSummary:")
print(f"{'Epsilon':<10}{'Base FGSM':<12}{'Base Filter':<14}{'Def FGSM':<12}{'Def Filter':<14}")
for i, eps in enumerate(epsilons):
    print(f"{eps:<10.2f}{fgsm_baseline[i]*100:<12.2f}{baseline_filter_accs[i]*100:<14.2f}{fgsm_defended[i]*100:<12.2f}{defended_filter_accs[i]*100:<14.2f}")