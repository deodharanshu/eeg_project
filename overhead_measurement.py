import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
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
y_train = torch.LongTensor(y_train)

n_epochs = 50
batch_size = 32
epsilon = 0.1

print("\n=== Standard training (baseline) ===")
model_std = EEGNet(n_classes=4, n_channels=22, n_timepoints=1001)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_std.parameters(), lr=0.001)

start_std = time.time()
for epoch in range(n_epochs):
    model_std.train()
    permutation = torch.randperm(X_train.size(0))
    
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_X = X_train[indices]
        batch_y = y_train[indices]
        
        optimizer.zero_grad()
        outputs = model_std(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

end_std = time.time()
time_std = end_std - start_std
print(f"Standard training time: {time_std:.2f} seconds ({time_std/60:.2f} min)")
print(f"Per-epoch: {time_std/n_epochs:.2f} seconds")

print("\n=== Adversarial training (defended) ===")
model_adv = EEGNet(n_classes=4, n_channels=22, n_timepoints=1001)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_adv.parameters(), lr=0.001)

start_adv = time.time()
for epoch in range(n_epochs):
    model_adv.train()
    permutation = torch.randperm(X_train.size(0))
    
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_X = X_train[indices]
        batch_y = y_train[indices]
        
        X_adv = fgsm_attack(model_adv, batch_X, batch_y, epsilon)
        X_mixed = torch.cat([batch_X, X_adv], dim=0)
        y_mixed = torch.cat([batch_y, batch_y], dim=0)
        
        optimizer.zero_grad()
        outputs = model_adv(X_mixed)
        loss = criterion(outputs, y_mixed)
        loss.backward()
        optimizer.step()

end_adv = time.time()
time_adv = end_adv - start_adv
print(f"Adversarial training time: {time_adv:.2f} seconds ({time_adv/60:.2f} min)")
print(f"Per-epoch: {time_adv/n_epochs:.2f} seconds")

overhead_ratio = time_adv / time_std
overhead_pct = (overhead_ratio - 1) * 100

print("\n=== Computational Overhead Summary ===")
print(f"Standard training:    {time_std:.2f} sec")
print(f"Adversarial training: {time_adv:.2f} sec")
print(f"Overhead ratio:       {overhead_ratio:.2f}x")
print(f"Overhead percentage:  +{overhead_pct:.1f}%")

print("\n=== Per-trial inference cost (negligible for both) ===")
model_std.eval()
n_inference_trials = 100
sample = X_train[:1]

start_inf = time.time()
with torch.no_grad():
    for _ in range(n_inference_trials):
        _ = model_std(sample)
end_inf = time.time()
inf_time_ms = (end_inf - start_inf) / n_inference_trials * 1000
print(f"Inference per trial: {inf_time_ms:.2f} ms (same for both, defense has no inference cost)")

with open("overhead_results.txt", "w") as f:
    f.write(f"Computational Overhead Measurement\n")
    f.write(f"==================================\n\n")
    f.write(f"Standard training (50 epochs):    {time_std:.2f} sec ({time_std/60:.2f} min)\n")
    f.write(f"Adversarial training (50 epochs): {time_adv:.2f} sec ({time_adv/60:.2f} min)\n")
    f.write(f"Per-epoch standard:    {time_std/n_epochs:.2f} sec\n")
    f.write(f"Per-epoch adversarial: {time_adv/n_epochs:.2f} sec\n")
    f.write(f"Overhead ratio: {overhead_ratio:.2f}x\n")
    f.write(f"Overhead percentage: +{overhead_pct:.1f}%\n\n")
    f.write(f"Inference per trial: {inf_time_ms:.2f} ms\n")
    f.write(f"Note: defense incurs zero inference cost. Same model architecture,\n")
    f.write(f"only training procedure differs. Critical for embedded BCI deployment.\n")
print("\nResults written to overhead_results.txt")