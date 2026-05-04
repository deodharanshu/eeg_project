from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

print("Downloading BCI Competition IV Dataset 2a...")
dataset = BNCI2014_001()
dataset.download()
print("Download complete.")

paradigm = MotorImagery(n_classes=4)
X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Classes: {set(y)}")
print("Dataset loaded successfully.")