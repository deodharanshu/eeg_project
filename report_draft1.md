# Threat Modelling and Defense Evaluation for EEG-Based Brain-Computer Interfaces

Anshuman Deodhar
ECE 547 Computer Security, Spring 2026
University of Massachusetts Amherst

---

## Methodology and Tools

This section walks you through the dataset, classifier, attacks, defense, and evaluation setup used in the project. The goal is to give you enough detail to recreate every result from scratch.

### Dataset

The project uses the BCI Competition IV Dataset 2a, a public motor imagery EEG dataset. The dataset records 9 subjects performing four-class motor imagery tasks: imagined movement of the left hand, right hand, both feet, and tongue. Each subject contributes two recording sessions, one for training and one for evaluation, with 288 trials per session and 576 trials per subject overall.

The signal acquisition uses 22 EEG channels sampled at 250 Hz. Each trial spans 4 seconds following cue onset. After preprocessing, every trial becomes a tensor of shape (22 channels, 1001 timepoints).

To keep iterative experiments fast, the project uses data from subjects 1 through 3, yielding 1728 trials. A stratified 80/20 split with a fixed random seed produces 1382 training trials and 346 test trials. The four classes stay balanced across both splits.

Data access goes through the MOABB library (Mother of All BCI Benchmarks), which handles raw EEG file formats and outputs standardized numpy arrays. This avoids manual handling of the GDF format and keeps preprocessing consistent across runs.

### Classifier Architecture

The target classifier is EEGNet, a compact convolutional neural network designed for EEG signal classification. Two reasons drove this choice. First, EEGNet is the standard benchmark in the EEG adversarial robustness literature, so the results in this project line up directly with prior published work. Second, its small parameter count matches the kind of model you would deploy on a resource-constrained embedded BCI device.

The architecture has two main convolutional blocks. The first block applies temporal filtering through a 1D convolution across the time axis, then a depthwise spatial convolution to learn spatial filters across the 22 channels. Batch normalization, ELU activation, average pooling, and dropout follow. The second block does separable convolution to capture temporal patterns at a coarser scale, with another round of pooling and dropout afterward. The output flattens through a fully connected layer to produce four class logits.

The full architecture lives in eegnet_model.py. Key hyperparameters: F1 equals 8 temporal filters, D equals 2 depth multiplier, F2 equals 16 separable filters, dropout rate 0.5.

### Baseline Training

The baseline EEGNet trains on clean unperturbed data using the Adam optimizer with learning rate 0.001, cross-entropy loss, batch size 32, and 50 epochs. No data augmentation runs. The trained model reaches 57.51% test accuracy on the 4-class task. This sits well above the 25% random-chance baseline and matches accuracy ranges reported in the EEGNet literature for this dataset.

The baseline model serves two purposes. The model is the white-box attack target throughout the project, and the model anchors the comparison against the defended version.

### Attack Implementations

The project implements four distinct attacks to cover different threat scenarios.

Fast Gradient Sign Method (FGSM). FGSM is the standard one-step gradient-based attack. Given a clean input X and its label y, the attack computes the gradient of the cross-entropy loss with respect to the input, takes its sign, scales by the perturbation budget epsilon, and adds the result to the original input. This produces an adversarial example X_adv equal to X plus epsilon times the sign of the input gradient. FGSM runs fast (one forward and backward pass per sample) but produces relatively crude perturbations. The implementation lives in fgsm_attack.py.

Projected Gradient Descent (PGD). PGD is the iterative version of FGSM and counts as the strongest first-order attack in the literature. Starting from the clean input plus small random noise inside the epsilon ball, PGD takes multiple gradient ascent steps with step size alpha equal to epsilon over 4, projecting back to the L-infinity epsilon ball after each step. The implementation runs 10 iterations. PGD is slower per attack but typically finds stronger adversarial examples than FGSM. The project includes PGD as a stress test for the defense: a defense holding under FGSM but breaking under PGD would suggest gradient masking rather than real robustness. The implementation lives in pgd_attack.py.

Black-box transfer attack. White-box attacks assume the attacker holds full access to the target model's weights and architecture. This rarely matches deployment reality. The black-box transfer attack tests the more realistic scenario where the attacker only has access to similar training data and has to train their own surrogate model. A separate ShallowCNN classifier (based on the architecture from Schirrmeister et al. 2017) trains on the same data, reaching 65.90% clean accuracy. FGSM adversarial examples then get generated using this surrogate and fed to the target EEGNet. If the attacks transfer well, the architecture difference offers no protection. If they fail to transfer, then keeping the target model weights private gives you meaningful security. The implementation lives in transfer_attack.py, with the surrogate architecture in shallow_cnn.py and its training in train_shallow.py.

Universal adversarial filter attack. This attack, based on Meng et al. 2024, is the most realistic of the four from a deployment standpoint. Instead of computing per-trial perturbations (which need gradient access for each new input at attack time), the attack learns a single set of filter coefficients. When you convolve those coefficients with any EEG signal, the classifier fails. The filter parameterizes as a depthwise 1D convolution with 22 channels and filter length 20. Training uses gradient ascent on the classification loss while the target model's weights stay frozen. Normalization scales the perturbation so its standard deviation equals epsilon times the signal standard deviation, with clipping at three times the budget to bound large outliers. Once trained, the filter applies to any test input as a single fixed transformation. This represents a far more deployable attack than per-trial FGSM. An attacker trains the filter once offline on their own data, then deploys the trained filter as a fixed signal-modification stage, possibly even at the analog hardware level. The implementation lives in adversarial_filter_attack.py.

### Defense Implementation

The defense is adversarial training. During training, every minibatch picks up FGSM adversarial examples generated against the current state of the model. The training loss runs on the concatenation of clean and adversarial examples with their original labels. The optimizer updates the model weights to classify both correctly, which forces the model to learn decision boundaries stable under small input perturbations.

The procedure: for each minibatch (X, y), compute X_adv as FGSM(model, X, y, epsilon equal to 0.1), form X_mixed as concat(X, X_adv) and y_mixed as concat(y, y), compute the loss on (X_mixed, y_mixed), and update the model weights with Adam. The defended model trains for 50 epochs with the same hyperparameters as the baseline. After training, the defended model reaches 57.80% clean accuracy, essentially matching the baseline's 57.51%. The defense imposes no clean accuracy cost. The implementation lives in adversarial_training.py.

### Evaluation Protocol

Both models (baseline and defended) face all four attacks at five perturbation budgets: epsilon in {0.01, 0.05, 0.10, 0.20, 0.30}. The metric is accuracy, defined as the fraction of test samples classified correctly. Comparison points are the 25% random-chance baseline (since this is a 4-class problem) and each model's clean accuracy.

For the universal filter attack, epsilon scales relative to the signal standard deviation rather than as an absolute amplitude. This puts the filter attack on a comparable scale with the other attacks given the actual signal magnitudes (signal standard deviation around 6.16 in the dataset units).

### Computational Overhead Measurement

To answer whether adversarial training is viable for embedded BCI deployment, the project measures training and inference times for both models. The standard training run is 50 epochs of standard EEGNet training. The adversarial training run is 50 epochs of the procedure described above. Inference timing repeats classification of a single trial 100 times and reports the mean. Measurements run on CPU only (no GPU acceleration) to reflect a constrained-hardware deployment. The implementation lives in overhead_measurement.py.

### Software Stack and Reproducibility

All code uses Python 3.12 with PyTorch 2.7.1 (CPU build), plus NumPy, scikit-learn, and Matplotlib for data handling and plots. The MOABB library handles dataset access. A fixed random seed (42) controls the train/test split to keep results reproducible. All code, trained model weights, and result plots live in a public Git repository at https://github.com/deodharanshu/eeg_project, with commit history reflecting the day-by-day progression of the project.

