# Threat Modelling and Defense Evaluation for EEG-Based Brain-Computer Interfaces

Anshuman Deodhar
ECE 547 Computer Security, Spring 2026
University of Massachusetts Amherst

---

[Abstract: 250 words. To be written last.]

---

## Introduction

### Motivation

Brain-computer interfaces (BCIs) translate neural signals into control commands for external devices. Electroencephalogram (EEG) is the most common input signal because EEG hardware is non-invasive, portable, and inexpensive. EEG-based BCIs are deployed in medical applications such as motor prostheses for paralyzed patients, communication tools for locked-in patients, neurorehabilitation for stroke recovery, and assessment of consciousness in unresponsive patients. Beyond medicine, EEG-based BCIs have also reached commercial markets for fatigue monitoring in drivers and operators.

Modern EEG-based BCIs rely on convolutional neural network classifiers to map raw EEG signals to user intent. These classifiers reach high accuracy and remove the need for hand-crafted feature engineering. The accuracy gains come with a security cost. Recent work shows EEG classifiers are vulnerable to adversarial examples. An attacker who crafts a small, almost imperceptible perturbation to the EEG signal forces the classifier to output the wrong command. In a wheelchair BCI, this means turning into traffic instead of around it. In a BCI speller for a locked-in patient, this means typing the wrong word, or worse, replacing the patient's intent with the attacker's. In a driver fatigue monitor, this means missing a microsleep before a crash. The stakes match medical device security in general: harm to a single patient, scaled across thousands of deployed devices.

### Background

The adversarial machine learning literature began in image classification. A small change to the pixels of a panda photo, invisible to the human eye, gets the image labeled as a gibbon. The same vulnerability applies to EEG classifiers. The first demonstration in BCIs was by Zhang and Wu in 2019, who showed a one-step gradient-based attack drops EEGNet accuracy from 80% to below random chance. Follow-up work extended these results to BCI spellers, regression problems like fatigue estimation, and a range of attack strategies including universal perturbations and signal-processing-based filter attacks.

Defenses against adversarial examples have a much shorter track record in BCIs than in image classification. The image-classification community has converged on adversarial training as the most reliable defense. Adversarial training augments the training set with adversarial examples generated against the model in the loop, forcing the model to learn decision boundaries stable under small perturbations. Whether this defense translates well to EEG is a separate empirical question, since EEG signals differ from images in dimensionality, noise structure, and inter-subject variability.

### Problem Statement

This project investigates two questions about the adversarial security of EEG-based BCIs:

First, how vulnerable is a standard EEG classifier (EEGNet trained on the BCI Competition IV 2a motor imagery dataset) to a range of adversarial attacks? The attacks considered cover three threat models in increasing realism: white-box gradient attacks (FGSM, PGD), black-box transfer attacks (FGSM examples generated on a different architecture and transferred to the target), and universal filter attacks (a learned signal-processing filter applied to any input).

Second, how well does adversarial training defend the classifier across these threat models, and at what cost? The defense is trained against FGSM perturbations only. The evaluation tests whether robustness generalizes to attack types the defense never saw, and whether the defense imposes any clean accuracy or computational cost relevant to embedded BCI deployment.

### Approach

The project trains a baseline EEGNet classifier on the BCI Competition IV 2a dataset, then attacks the trained model under each of the four threat scenarios above. The same classifier is then retrained with FGSM adversarial training and re-evaluated under all attacks. Computational overhead (training time and inference time) is measured to address deployment viability. Results are presented as accuracy versus perturbation magnitude curves, comparing the defended and undefended models against random chance.

### Expected Results

The expected results follow patterns reported in prior EEG adversarial security literature. White-box attacks should drop accuracy below random chance at moderate perturbation budgets. Adversarial training should restore substantial robustness against the white-box attack the defense was trained on. The open questions are whether the defense holds against stronger iterative attacks (PGD), whether the defense generalizes to structurally different attacks (universal filter), and whether black-box transfer attacks behave like the image-classification literature predicts (transferring well across architectures) or differently for EEG.

### Connection with the Rest of the Course

The project applies several themes from ECE 547 directly. From lecture 11 on medical device security, the project follows the threat-modeling framework of asset identification, adversary motivation, adversary resources, and attack vectors. The asset under attack is the integrity of the EEG-to-command mapping in the BCI, which translates to user safety and intent. Adversary motivations for BCI attacks span violence (driving a wheelchair into traffic), fraud (manipulating BCI-based authentication or BCI-controlled financial transactions), and harassment or coercion (forcing the user to type unintended messages through a BCI speller). Adversary resources range from individuals with access to off-the-shelf machine learning tools (sufficient for transfer-based black-box attacks) to organizations capable of compromising the model weights or signal-processing chain (sufficient for white-box and universal-filter attacks).

The project also applies eight design principles from the same lecture: incorporating security early, encrypting sensitive data, authenticating data sources, using well-studied cryptographic primitives, avoiding security through obscurity, source-code analysis, building a realistic threat model, and side-channel protection. The principle most directly tested in this project is the realistic threat model. The black-box transfer experiment in this project tests whether the realistic-attacker threat model (no model weight access) is meaningfully weaker than the worst-case white-box threat model.

Beyond lecture 11, the project draws on broader ECE 547 themes around machine learning security, signal-processing-level attack vectors, and the practical-attacker mindset distinguishing deployable security claims from idealized ones.

### Organization of the Rest of the Report

The remainder of the report follows the structure below. The Related Work section reviews the four papers central to this project: the EEGNet architecture, the first adversarial attack study on EEG-based BCIs, the universal filter attack, and the adversarial defense benchmark for EEG classifiers. The Methodology and Tools section walks through the dataset, classifier, attacks, defense, and evaluation setup in enough detail for a reader to recreate the experiments. The Partitioning of Work and Schedule section documents the day-by-day project timeline. The Experiments and Results section presents all numerical results, comparison plots, and the consolidated summary table. The Analysis and Reflections section interprets the findings, discusses what additional work would have improved the project, and identifies skill gaps. The Conclusions and Future Work section closes with the project's contributions and open directions.

---

## Related Work

This section reviews the four papers most directly relevant to the project. Each paper contributes a specific element: the target classifier architecture, the attack taxonomy and threat model framing, a practical attack vector, and a defense benchmark on the same problem domain.

### EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain-Computer Interfaces (Lawhern et al. 2018)

EEGNet is the target classifier for every experiment in this project. Lawhern and colleagues designed EEGNet as a compact convolutional architecture specifically for EEG signal classification. Two design choices distinguish EEGNet from generic CNNs. First, the architecture decomposes the convolution into two stages: a temporal convolution learning frequency-band filters, followed by a depthwise spatial convolution learning frequency-specific spatial filters across EEG channels. This decomposition mirrors the filter-bank common spatial pattern algorithm, a long-standing technique in EEG signal processing, but learned end-to-end. Second, the architecture uses depthwise and separable convolutions throughout, keeping the parameter count under 2,000 for typical configurations. The original paper validates EEGNet on four BCI paradigms (P300, ERN, MRCP, and SMR motor imagery) and shows the architecture generalizes across paradigms with limited training data.

EEGNet matters for this project for three reasons. The compactness makes the architecture representative of classifiers you would deploy on resource-constrained embedded BCI hardware, which keeps the security findings deployment-relevant rather than theoretical. The widespread adoption of EEGNet in the EEG adversarial robustness literature makes the results in this project directly comparable to prior published work. The depthwise separable convolution design also matters for the universal filter attack, since the filter-and-spatial-filter structure of EEGNet aligns with how the Meng et al. 2024 attack perturbs the signal.

### On the Vulnerability of CNN Classifiers in EEG-Based BCIs (Zhang and Wu 2019)

Zhang and Wu's paper is the foundational adversarial security study for EEG-based BCIs. The paper proposes an attack framework based on injecting a jamming module between signal preprocessing and machine learning, then introduces an unsupervised version of the Fast Gradient Sign Method (UFGSM) needing no ground-truth labels at attack time. The paper covers three threat scenarios this project mirrors directly: white-box attack (attacker has full model access), gray-box attack (attacker has training data but not model parameters), and black-box attack (attacker has only query access to the deployed model). The experiments cover three CNN classifiers (EEGNet, DeepCNN, ShallowCNN) on three datasets (P300, ERN, motor imagery).

The findings most relevant to this project: white-box attacks drop EEGNet accuracy from above 65% to below 25% at small perturbation budgets, gray-box attacks perform less effectively than white-box but still cause large drops, and black-box attacks via substitute models also succeed though less reliably. The paper also notes adversarial perturbations are barely visible in the time-domain EEG signal, matching the canonical adversarial example phenomenon from image classification.

This project follows Zhang and Wu's threat-model taxonomy exactly. The white-box FGSM and PGD attacks in this project test the worst-case scenario the paper studies. The black-box transfer attack via a separate ShallowCNN surrogate tests the realistic deployment threat the paper raises but treats as substantially weaker. The motor imagery dataset and EEGNet classifier match the paper's setup, which keeps the baseline comparisons direct.

### Adversarial Filtering Based Evasion and Backdoor Attacks to EEG-Based Brain-Computer Interfaces (Meng et al. 2024)

Meng and colleagues introduce a different attack mechanism than gradient-based methods. Instead of computing per-trial perturbations, the attack learns a single adversarial filter once, offline, on training data. Once trained, the filter applies as a fixed signal transformation to any input EEG trial. The paper formulates the attack as an optimization problem balancing attack effectiveness (high cross-entropy loss on the target classifier) against attack stealthiness (low mean-squared-error distortion of the filtered signal). Binary search on the trade-off parameter finds the balance dropping classifier accuracy to chance level while keeping signal distortion small enough to evade detection.

The attack targets the signal-processing stage of the BCI rather than the model itself, allowing deployment as a hardware-level signal modifier rather than a digital injection. The paper reports attack success rates above 90% on three CNN classifiers across three EEG datasets. The filter also transfers across architectures, which makes the attack usable in a black-box scenario where the attacker trains the filter on a substitute model.

This project implements the universal filter attack from Meng et al. 2024 against the trained EEGNet baseline and the adversarially-trained defended model. The implementation uses a depthwise 1D convolution with 22 channels (matching the dataset) and length 20, trained by gradient ascent on the classification loss with the target model frozen. The results extend Meng et al.'s findings by evaluating whether adversarial training, which was not the focus of the original paper, generalizes as a defense against this structurally different attack type.

### Adversarial Robustness Benchmark for EEG-Based Brain-Computer Interfaces (Meng et al. 2023)

The 2023 paper by the same group benchmarks nine adversarial defense approaches against multiple attacks on three CNN classifiers and two EEG datasets. The defenses span four categories: robust training (adversarial training, TRADES, shift-consistency regularization), adversarial pruning (HYDRA, stochastic activation pruning), input transformation (shifting, sampling, channel shuffling, amplitude scaling, Gaussian noise addition, random transformation), and model ensemble (random self-ensemble, self-ensemble adversarial training). The paper reports results across white-box and black-box attacks under both L-infinity and L-2 norm constraints, in both within-subject and cross-subject evaluation settings.

The findings most relevant to this project: adversarial training and its variants achieve the strongest robustness against white-box attacks across all tested datasets and classifiers. The defense does cost some clean accuracy, but the trade-off is acceptable for the gain in robustness. Robust training combined with pruning (HYDRA) achieves comparable robustness to standalone robust training while compressing the model, which matters for embedded deployment. Stochastic activation pruning achieves the best robustness against AutoAttack, an approach combining multiple attack strategies. Input transformations show mixed results, with some (Gaussian noise addition, sampling) improving both clean accuracy and robustness simultaneously.

This project implements one defense from this benchmark, FGSM-based adversarial training, applied to EEGNet on the motor imagery dataset. The choice of adversarial training over the more sophisticated TRADES or HYDRA reflects the project scope: a single-person undergraduate project covering attack and defense evaluation in a limited time window, rather than a comprehensive defense comparison. The experimental results in this project replicate the benchmark's finding: FGSM adversarial training provides meaningful robustness without significant clean accuracy cost on EEG data. The benchmark also serves as a reference point for what stronger defenses would have achieved, a topic the Analysis and Reflections section returns to.


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

---

## Experiments and Results

This section walks you through every experiment in the project. Each subsection covers one piece of the picture: clean performance, white-box attacks, the defense, defense generalization, black-box transferability, the universal filter attack, and computational cost. A summary table at the end consolidates all numbers in one place.

### Baseline Performance

Before any attack runs, the three trained models reach the following clean accuracies on the held-out test set of 346 trials:

The baseline EEGNet reaches 57.51% test accuracy. The defended EEGNet reaches 57.80%. The ShallowCNN surrogate reaches 65.90%. All three sit well above the 25% random-chance baseline for the 4-class problem, which confirms each model learned meaningful motor imagery features. The defended model's clean accuracy slightly exceeds the baseline, which suggests adversarial training works as a mild regularizer in this setting and imposes no clean accuracy penalty.

The ShallowCNN's higher clean accuracy than EEGNet matters for the black-box experiment later. A competent surrogate produces stronger transferable attacks than an underfit one, so this number sets up a fair black-box stress test.

### White-box Attack Results

The FGSM attack runs against the baseline EEGNet across five perturbation budgets. Results below show classification accuracy as epsilon increases:

Epsilon 0.01: 52.02%
Epsilon 0.05: 35.26%
Epsilon 0.10: 16.76%
Epsilon 0.20: 2.02%
Epsilon 0.30: 0.00%

At epsilon equal to 0.10, accuracy already drops below the 25% random-chance line, meaning the attacker does better by flipping the model's prediction than by random guessing. At epsilon equal to 0.30, every single test trial gets misclassified. The classifier produces zero correct predictions under attack.

PGD with 10 iterations runs against the same baseline model. PGD produces almost identical results to FGSM across all epsilons (52.02%, 34.97%, 15.61%, 1.73%, 0.00%). The accuracy gap between PGD and FGSM stays under 2 percentage points at every epsilon. This finding deviates from typical image classification results, where PGD usually beats FGSM substantially. EEG signals likely have lower effective decision dimensionality than images, so the single-step gradient direction from FGSM already captures most of the available perturbation power and additional iterations produce diminishing returns.

Figure 1 visualizes the FGSM attack on the baseline. The left panel shows accuracy versus epsilon, and the right panel shows an example clean and adversarial EEG trace at epsilon equal to 0.10. The two signals look nearly identical to the human eye, yet the classifier produces wrong outputs on the adversarial version. This is the canonical adversarial example phenomenon: imperceptibly small perturbations cause complete model failure.

[Insert Figure 1: fgsm_attack_results.png]

### Defense Effectiveness

The adversarial training defense uses FGSM examples at epsilon equal to 0.10 mixed with clean examples during training. The defended model gets evaluated against FGSM at the same five epsilons:

Epsilon 0.01: 56.36%
Epsilon 0.05: 47.98%
Epsilon 0.10: 37.86%
Epsilon 0.20: 20.52%
Epsilon 0.30: 10.40%

Compare these against the undefended baseline. At epsilon equal to 0.10, the defense holds 37.86% accuracy versus the baseline's 16.76%. The defense more than doubles robustness at the training epsilon. At epsilon equal to 0.30, the defended model retains 10.40% accuracy while the baseline collapses to 0%. Defense quality degrades at higher epsilons (since training only used epsilon equal to 0.10), but the model never fully breaks.

The defense does not produce invincibility. The defended model still loses accuracy under attack, only far less than the baseline. The right framing: adversarial training shifts the entire accuracy-versus-epsilon curve upward without requiring any clean accuracy tradeoff.

Figure 2 plots both curves together. The gap between defended and undefended widens with epsilon, which shows the defense matters most at the perturbation magnitudes where the undefended model fails completely.

[Insert Figure 2: defense_comparison.png]

### Defense Generalization to PGD

A defense trained on FGSM might learn a narrow trick (gradient masking) rather than real robustness. To test for this, PGD with 10 iterations runs against the defended model. The results match the FGSM defended results to within 1.5 percentage points at every epsilon:

Epsilon 0.01: 56.36% under FGSM, 56.36% under PGD
Epsilon 0.05: 47.98% under FGSM, 47.98% under PGD
Epsilon 0.10: 37.86% under FGSM, 37.86% under PGD
Epsilon 0.20: 20.52% under FGSM, 20.23% under PGD
Epsilon 0.30: 10.40% under FGSM, 8.96% under PGD

The defense holds against the stronger iterative attack as well as the defense holds against the one-step attack. This rules out gradient masking. The robustness generalizes to attack types the defense did not train on, which means the defense provides real (not illusory) protection.

Figure 3 shows the PGD comparison side by side with the FGSM comparison.

[Insert Figure 3: pgd_comparison.png]

### Black-box Transferability

The black-box scenario assumes the attacker has no access to the target model weights and trains a separate ShallowCNN surrogate on the same training data. FGSM examples generated against the surrogate then transfer to the EEGNet target. Results below show accuracy of the baseline EEGNet under transferred attacks:

Epsilon 0.01: 56.07%
Epsilon 0.05: 54.05%
Epsilon 0.10: 50.29%
Epsilon 0.20: 40.75%
Epsilon 0.30: 32.95%

Compare against white-box FGSM on the same baseline model: 52.02%, 35.26%, 16.76%, 2.02%, 0.00%. At every epsilon, transferred black-box attacks lose substantial power. At epsilon equal to 0.10, the white-box attack drops accuracy to 16.76% but the black-box attack only drops accuracy to 50.29%. At epsilon equal to 0.30, white-box reaches 0% but black-box stops at 32.95%.

This deviates sharply from image classification, where adversarial examples notoriously transfer well across architectures. EEG signals appear to produce more model-specific decision boundaries, so attacks crafted against one architecture fail to break another.

The defended model holds up even better against black-box transfer. At epsilon equal to 0.30, the defended model retains 41.04% accuracy under transferred attack versus 10.40% under white-box FGSM. Adversarial training's robustness covers attack types beyond the one used during training.

Figure 4 plots white-box versus black-box for both models.

[Insert Figure 4: transfer_attack_results.png]

The deployment implication: white-box attacks are devastating, but black-box attacks face a real architecture barrier on EEG. Protecting model weights matters. If an attacker only has training data and not the model weights, the threat drops substantially.

### Universal Filter Attack

The universal adversarial filter (based on Meng et al. 2024) is the most realistic attack vector against deployed BCIs. Instead of computing per-trial gradients (which require the attacker to access the model at inference time), the filter trains once offline. The trained filter applies as a fixed signal transformation to any input. Results against the baseline EEGNet:

Epsilon 0.01: 55.20%
Epsilon 0.05: 49.42%
Epsilon 0.10: 40.75%
Epsilon 0.20: 26.59%
Epsilon 0.30: 33.24%

The filter produces meaningful attacks at higher epsilons (the model drops below random chance at epsilon equal to 0.20). The slight uptick at epsilon equal to 0.30 reflects training noise: the filter at this epsilon converged to a slightly less effective solution. The filter remains weaker per epsilon than per-trial FGSM (the universality cost: one filter must work across all trials, so cannot tune to individual examples), but the filter still attacks effectively at higher budgets.

The defended model holds up even better against the filter than against FGSM. At epsilon equal to 0.10, defended-filter retains 45.38% versus defended-FGSM at 37.86%. At epsilon equal to 0.20, defended-filter retains 37.28% versus defended-FGSM at 20.52%. The defense generalizes to a structurally different attack type.

Figure 5 plots the filter attack against per-trial FGSM for both models.

[Insert Figure 5: filter_attack_results.png]

The filter attack matters for deployment. A real attacker against a BCI has no easy path to compute per-trial gradients on an embedded device. The attacker trains a filter once on their own setup and deploys the filter as a hardware-level signal modifier. The filter results show this realistic attack works, though less effectively than the worst-case white-box scenario.

### Computational Overhead

A defense doubling training time but keeping inference cheap is deployment-viable. A defense slowing inference is not. Measurements ran on CPU only (no GPU acceleration) to reflect constrained-hardware deployment.

Standard EEGNet training (50 epochs): 201.84 seconds (3.36 minutes)
Adversarial training (50 epochs): 1056.47 seconds (17.61 minutes)
Training overhead ratio: 5.23x
Inference time per trial: 4.06 ms (identical for both models)

The 5.23x training overhead is higher than the 2-3x typical for adversarial training, which reflects CPU-only execution and the doubled batch size during training (clean and adversarial examples concatenated). On GPU hardware, the overhead drops closer to 2-3x because the larger batch parallelizes well.

The inference time matters more than the training time for deployment. Adversarial training does not change the model architecture. Both models run the same forward pass at inference. The defended model has zero additional inference cost. At 4.06 milliseconds per trial, both models classify around 246 trials per second on CPU, well within real-time BCI requirements (most BCIs target latencies under 100 ms per classification).

Training is an offline cost paid once on a workstation. Inference is the online cost paid on the deployed device. Adversarial training shifts cost into the offline phase only, which keeps the defense deployment-viable for embedded BCIs.

### Summary Table

The table below consolidates classification accuracy across all attacks, both models, and all epsilons.

Clean accuracy: Baseline 57.51%, Defended 57.80%, ShallowCNN surrogate 65.90%

Epsilon 0.01: Baseline FGSM 52.02%, Baseline PGD 52.02%, Baseline Black-box 56.07%, Baseline Filter 55.20%, Defended FGSM 56.36%, Defended PGD 56.36%, Defended Black-box 56.65%, Defended Filter 56.94%

Epsilon 0.05: Baseline FGSM 35.26%, Baseline PGD 34.97%, Baseline Black-box 54.05%, Baseline Filter 49.42%, Defended FGSM 47.98%, Defended PGD 47.98%, Defended Black-box 56.07%, Defended Filter 54.62%

Epsilon 0.10: Baseline FGSM 16.76%, Baseline PGD 15.61%, Baseline Black-box 50.29%, Baseline Filter 40.75%, Defended FGSM 37.86%, Defended PGD 37.86%, Defended Black-box 52.02%, Defended Filter 45.38%

Epsilon 0.20: Baseline FGSM 2.02%, Baseline PGD 1.73%, Baseline Black-box 40.75%, Baseline Filter 26.59%, Defended FGSM 20.52%, Defended PGD 20.23%, Defended Black-box 48.27%, Defended Filter 37.28%

Epsilon 0.30: Baseline FGSM 0.00%, Baseline PGD 0.00%, Baseline Black-box 32.95%, Baseline Filter 33.24%, Defended FGSM 10.40%, Defended PGD 8.96%, Defended Black-box 41.04%, Defended Filter 34.39%

Three patterns hold across the table. First, white-box attacks (FGSM, PGD) are devastating against the undefended baseline and substantially weakened against the defended model. Second, attack difficulty for the attacker scales with realism: white-box is easiest for the attacker and most damaging, black-box transfer is hardest and least damaging, and the universal filter sits in between. Third, the defended model improves robustness against every attack type at every epsilon, with no clean accuracy cost.


---

## Analysis and Reflections

### What I Learned

The most concrete takeaway from the project is the threat-model taxonomy from Zhang and Wu 2019. Before this project, I treated adversarial machine learning as a single concept: an attacker breaks a model. The taxonomy splits the threat into white-box, gray-box, and black-box scenarios based on what the attacker knows, and each scenario implies different attack strategies, different defense priorities, and different real-world likelihoods. The black-box transfer experiment in this project drove the lesson home: the white-box result (model accuracy drops to 0%) is genuinely scary in isolation, but the black-box result (accuracy stays above 30% even at large perturbations) reframes the whole picture. White-box devastation is only achievable by an attacker who already has the model weights, which is itself a serious security failure independent of adversarial examples. The realistic threat is the black-box transfer attack, and the realistic threat is much weaker than the worst case the literature usually presents.

The second takeaway is the universality cost trade-off in adversarial attacks. Per-trial FGSM achieves 0% target accuracy at large perturbations. The universal filter, which is the practical-deployment version of the attack, only achieves around 33% at the same budget. The gap between worst-case and deployable attack reflects a real engineering trade-off the literature does not emphasize enough. An attacker building a real-world attack against a BCI has no easy way to compute per-trial gradients at deployment time. The attacker has to pre-compute a universal perturbation, which sacrifices effectiveness for practicality. This pattern likely extends beyond BCIs to any deployed ML system facing realistic attackers.

The third takeaway is more methodological: adversarial training works as a defense, but the gain shifts the entire accuracy-versus-epsilon curve upward rather than producing immunity. The defended model still loses accuracy under attack. The right framing for defense effectiveness is not "did the attack fail" but "by how much did the attack get weaker." The benchmark from Meng et al. 2023 reinforces this framing across nine defenses and multiple attacks.

A fourth takeaway came from debugging the universal filter attack. The first working version of the attack barely affected the model: filtered inputs dropped accuracy by only 1% even at large perturbation budgets. Tracing the issue revealed two compounding problems. The perturbation normalization scaled by the maximum absolute value across the entire batch, which meant 99.9% of perturbation values got scaled toward zero while only the single peak value reached the budget. The filter coefficients were also initialized far below the signal scale, which produced near-zero gradients. Fixing both required rescaling the perturbation budget relative to the signal standard deviation and increasing the filter initialization variance by an order of magnitude. The lesson generalizes: adversarial attack implementations are sensitive to scale assumptions, and a quiet failure (attack runs without errors but produces no perturbation) looks identical to a successful defense in the result table. Verifying an attack genuinely attacks is as important as evaluating the defense.

### What One More Week Would Have Bought

With one more week, the obvious target is the gap items from the proposal. The threat model section grounded in Burleson's lecture 11 framework is in this report, but only at the level of a connection paragraph. A full week would let me apply the framework concretely: walk through the asset identification for a deployed BCI, enumerate adversary types with specific motivations and resource levels, map each of the four implemented attacks to a specific adversary profile, and discuss which of Burleson's eight design principles each attack violates or exploits. The current report only gestures at this analysis.

A second target is broader experimental coverage. The current evaluation uses three subjects from the BCI Competition IV 2a dataset. With one more week, I would extend the evaluation to all nine subjects and look at cross-subject generalization. The attack and defense behavior likely varies meaningfully across subjects, and the current single-split evaluation misses this variability. The Meng et al. 2023 benchmark reports both within-subject and cross-subject results, and the gap between the two is large for some defenses, which would change the conclusions of this project.

A third target is implementing one stronger defense from the Meng et al. 2023 benchmark, likely TRADES or HYDRA. The current report uses FGSM adversarial training because the approach is the simplest defense to implement and to explain. The benchmark suggests TRADES achieves similar robustness with better clean-accuracy preservation, and HYDRA combines robustness with model compression for embedded deployment. Both would make the defense story more complete.

### What One More Month Would Have Bought

With one more month, the natural extension is to move beyond the BCI Competition IV 2a dataset to a different EEG paradigm. Motor imagery is a relatively easy classification problem with strong, well-localized features. The interesting security question is whether the findings in this project generalize to harder paradigms like P300 spellers or driver fatigue regression, where the signal-to-noise ratio is lower and the decision boundaries are more diffuse. Zhang and Wu 2019 covers P300 in passing, but their analysis is brief, and a focused study on whether universal filter attacks behave differently on P300 versus motor imagery would be a publishable contribution.

A second one-month target is implementing the backdoor attack from Meng et al. 2024, which the same paper introduces alongside the universal filter evasion attack. Backdoor attacks are a different threat class entirely: the attacker poisons the training data to install a hidden trigger, then activates the trigger at deployment time. The threat model is much more dangerous than evasion attacks because the attacker does not need ongoing access to the deployed system. Defending against backdoor attacks in EEG-based BCIs is largely unexplored territory.

A third one-month target is broader threat-model analysis combined with multidisciplinary review. The proposal mentions integrating ethics, economics, and policy considerations into the threat model. The current report does not do this. A serious treatment would discuss who funds BCI deployment (insurance companies, the patient, hospital systems), what regulatory framework applies (FDA medical device approvals, GDPR for EEG data), what economic incentives push BCI companies toward or away from security investment, and what bioethical concerns arise when adversarial attacks would manipulate the intentions of patients with disorders of consciousness. This analysis sits closer to the policy side of computer security than the technical side, but Burleson's lecture 11 framing makes clear both sides matter for medical device security.

### Skills Gaps and Useful Background

The most directly useful skill would have been deeper PyTorch fluency. EEG adversarial security work involves frequent custom modifications to training loops (mixing clean and adversarial batches, freezing model weights while training filter parameters, projecting perturbations into bounded balls). Each of these requires comfort with PyTorch's autograd mechanics, parameter registration, and the difference between in-place and out-of-place tensor operations. The current project taught the patterns by working through them, but a stronger foundation in PyTorch from a full deep learning course would have made the experimental cycle faster and the implementation choices more independent.

A second skill gap is in EEG signal processing specifically. The project uses pre-built tools (MOABB for dataset access, EEGNet for the classifier) hiding most of the signal-processing detail. A stronger EEG background would have helped in interpreting the spectral analysis from the Zhang and Wu 2019 paper, understanding why the filter-bank common spatial pattern algorithm matters as inspiration for EEGNet's architecture, and reasoning about which frequency components the universal adversarial filter targets. The current project treats EEG signals as generic time-series data, which works but loses domain-specific insight.

A third skill gap is in the security-policy side of the threat-model analysis. Burleson's medical device security framework draws on his own DAC 2012 paper and on broader medical-device-security literature I had limited exposure to before this course. A stronger background in the security-policy literature, especially around medical device regulation and the FDA approval process, would have made the threat-modeling discussion in the introduction far richer. The current treatment connects the project to lecture 11 at the level of a competent summary rather than at the level of original analysis.


---

## Conclusions and Future Work

### Conclusions

This project evaluated the adversarial security of an EEG-based brain-computer interface classifier under four attack scenarios and one defense. The target classifier was EEGNet trained on the BCI Competition IV 2a motor imagery dataset, reaching 57.51% baseline accuracy on a 4-class task.

The experimental findings break down across four results. First, white-box gradient attacks devastate the undefended classifier. FGSM at perturbation budget 0.10 drops accuracy from 57.51% to 16.76%, below the 25% random-chance baseline. At perturbation budget 0.30, accuracy drops to 0%. The classifier produces zero correct predictions under attack. PGD with 10 iterations matches FGSM closely, with accuracy gaps under 2 percentage points across all perturbation budgets.

Second, FGSM-based adversarial training works as a defense without imposing clean accuracy cost. The defended model reaches 57.80% clean accuracy, slightly above the undefended baseline. Under FGSM attack at perturbation 0.10, the defended model holds 37.86% accuracy versus the undefended 16.76%. The defense generalizes to PGD almost perfectly: defended accuracy under PGD matches defended accuracy under FGSM to within 1.5 percentage points. The defense is not memorizing the training-time attack.

Third, black-box transfer attacks are substantially weaker than white-box attacks for EEG classifiers. FGSM examples generated on a separate ShallowCNN surrogate transfer poorly to the EEGNet target. At perturbation 0.10, transferred attacks drop baseline accuracy only to 50.29% versus the white-box drop to 16.76%. The defended model holds even better against transfer attacks, reaching 41.04% accuracy at perturbation 0.30 where white-box FGSM drove the same model to 10.40%. This is the opposite pattern from image classification, where adversarial examples typically transfer well across architectures.

Fourth, the universal filter attack from Meng et al. 2024 is weaker than per-trial gradient attacks but more deployable. The filter drops baseline accuracy to around 33% at perturbation 0.30 (vs 0% for FGSM), reflecting the universality cost. The defense holds even better against the filter than against FGSM: defended-filter accuracy at perturbation 0.20 is 37.28% versus defended-FGSM at 20.52%.

Fifth, the defense is deployment-viable for embedded BCIs. Adversarial training takes 5.23 times longer than standard training (17.61 minutes vs 3.36 minutes for 50 epochs on CPU), but inference cost is identical: 4.06 milliseconds per trial for both models. Training is an offline cost paid once on a workstation. Inference is the online cost paid on the deployed device. The defense imposes no inference-time penalty.

The combined picture reframes the security threat to EEG-based BCIs. White-box attacks are catastrophic but require model weight access, which is itself a serious breach. Black-box attacks via realistic substitute models are much weaker, and the gap to white-box devastation is large. Adversarial training closes most of the remaining vulnerability at no inference cost. The deployable threat to a properly-secured BCI is meaningfully smaller than the white-box literature alone suggests.

### Future Work

Several directions extend the work in this project. The most direct extension is full-dataset evaluation across all nine subjects with cross-subject generalization analysis. The current within-subject evaluation on three subjects misses the variability mattering for real deployment.

A second extension is implementing stronger defenses from the Meng et al. 2023 benchmark. TRADES provides a principled trade-off between clean accuracy and robustness. HYDRA combines robust training with model compression, which fits the embedded BCI deployment context. Stochastic activation pruning achieves the strongest robustness against AutoAttack in the benchmark, and would test whether this project's findings hold against the strongest current attack ensemble.

A third extension is the backdoor attack from Meng et al. 2024. Backdoor attacks poison the training set to install a hidden trigger, which is a fundamentally different threat from the evasion attacks studied in this project. Defending against backdoor attacks in EEG-based BCIs is largely open, and would require modifications to the data pipeline and the training procedure rather than the model architecture.

A fourth extension is moving beyond the motor imagery paradigm to harder BCI tasks. P300 spellers, error-related negativity detection, and continuous regression tasks like driver fatigue estimation all have different signal characteristics. Whether the universality cost trade-off and the black-box weakness pattern hold across paradigms is an open empirical question.

A fifth extension is a serious treatment of the multidisciplinary aspects of BCI security: regulatory frameworks for medical-device approval, economic incentives shaping security investment, and bioethical concerns specific to neural interfaces. The current project covers the technical side. The complete picture also covers who pays for BCI security, who suffers when the security fails, and what legal recourse a harmed patient has.

