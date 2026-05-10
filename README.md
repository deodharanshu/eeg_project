# Adversarial Security of EEG-Based Brain-Computer Interfaces

End-to-end evaluation of attack vectors and defense effectiveness for a deep learning EEG classifier deployed in a brain-computer interface (BCI) setting. Implements four adversarial attack types spanning three threat models, one defense, and a deployment-viability analysis.

Course project for ECE 547 Computer Security, UMass Amherst, Spring 2026 (Prof. Wayne Burleson).

## What This Project Demonstrates

EEG-based BCIs translate neural signals into device commands using convolutional neural networks. Deployments include motor prostheses, communication devices for locked-in patients, and clinical fatigue monitors. Recent research shows these classifiers are vulnerable to adversarial examples: small perturbations that force misclassification. The patient safety stakes make this a high-priority security domain.

This project answers two questions: how vulnerable is a standard EEG classifier to a realistic range of attacks, and how well does adversarial training defend the classifier across attack types it never saw during training?

## Key Findings

| Result | Number |
|---|---|
| Baseline EEGNet clean accuracy (9 subjects) | 56.22% |
| Baseline accuracy under FGSM at epsilon = 0.10 | 13.40% |
| Defended accuracy under FGSM at epsilon = 0.10 | 36.26% |
| Defended accuracy under PGD at epsilon = 0.10 | 36.07% |
| Defense clean accuracy cost | 1.06 percentage points |
| Black-box transfer attack accuracy at epsilon = 0.10 | 50.29% |
| Training overhead (defense vs baseline) | 2.50x |
| Inference overhead | 0% |

Five conclusions follow from the experiments:

1. White-box gradient attacks devastate the undefended classifier, dropping accuracy below 1% at moderate perturbation budgets.
2. Adversarial training nearly triples robustness at the training epsilon with only a 1.06 percentage point clean accuracy cost.
3. The defense generalizes to PGD almost perfectly without being trained on PGD examples, ruling out gradient masking.
4. Black-box transfer attacks are substantially weaker than white-box attacks for EEG classifiers, contradicting the canonical pattern from image classification.
5. The defense imposes 2.50x training overhead but zero inference overhead, making it deployment-viable for embedded BCIs.

## Threat Model Coverage

The project follows the white-box / gray-box / black-box taxonomy from Zhang and Wu 2019, plus the signal-processing-level filter attack from Meng et al. 2024.

| Attack | Threat Model | Adversary Capability |
|---|---|---|
| FGSM (Goodfellow 2014) | White-box | Full model weights access |
| PGD (Madry 2018) | White-box | Full model weights access |
| Black-box transfer | Black-box | Public dataset and a substitute model architecture |
| Universal adversarial filter (Meng 2024) | Signal-processing-level | Hardware-level signal modification, no model access |

## Tech Stack

- **Framework:** PyTorch 2.7.1 (CPU)
- **Dataset:** BCI Competition IV Dataset 2a (motor imagery, 9 subjects, 22 channels, 4 classes)
- **Dataset access:** MOABB
- **Target classifier:** EEGNet (Lawhern 2018)
- **Surrogate classifier (for black-box attacks):** ShallowCNN (Schirrmeister 2017)

## Repository Structure

```
.
├── eegnet_model.py            # EEGNet architecture
├── shallow_cnn.py             # ShallowCNN surrogate architecture
├── download_data.py           # Dataset acquisition via MOABB
├── train_baseline.py          # Train clean EEGNet
├── train_shallow.py           # Train ShallowCNN surrogate
├── adversarial_training.py    # Defense: FGSM-based adversarial training
├── fgsm_attack.py             # White-box FGSM attack
├── pgd_attack.py              # White-box PGD attack with iterative descent
├── transfer_attack.py         # Black-box FGSM transfer via surrogate
├── adversarial_filter_attack.py   # Universal filter attack (Meng 2024)
├── overhead_measurement.py    # Training and inference timing
├── full_subject_evaluation.py # 9-subject full pipeline
├── make_separate_plots.py     # Result plotting
├── fix_fgsm_figure.py         # Figure regeneration scripts
├── fix_filter_figure.py
├── *.png                      # Result figures
├── *.pth                      # Trained model weights
├── *.txt                      # Experiment logs
└── report_draft*.md           # Project report (draft progression)
```

## Running the Experiments

```bash
# Setup
pip install torch numpy moabb scikit-learn matplotlib

# Step 1: Download the dataset (cached after first run)
python download_data.py

# Step 2: Train the baseline classifier
python train_baseline.py

# Step 3: Run the FGSM attack on the baseline
python fgsm_attack.py

# Step 4: Train the adversarially-trained defended classifier
python adversarial_training.py

# Step 5: Run all attacks on both models (9-subject scope)
python full_subject_evaluation.py

# Step 6: Run black-box transfer attack (3-subject scope, takes longer)
python train_shallow.py
python transfer_attack.py

# Step 7: Run universal filter attack (3-subject scope, slowest)
python adversarial_filter_attack.py

# Step 8: Measure computational overhead
python overhead_measurement.py
```

Total compute time on CPU: roughly 2 hours for the 9-subject pipeline, plus 1-2 hours for the black-box and filter experiments at 3-subject scope.

## Scope Notes

The baseline, defense, FGSM, and PGD experiments use all 9 subjects (5184 trials, 4147/1037 train/test split). The black-box transfer and universal filter attack experiments use a 3-subject subset (1728 trials) due to higher per-experiment compute cost. The split scope is documented explicitly throughout the report.

## References

The four primary papers grounding this project:

1. Zhang, X. and Wu, D. (2019). "On the Vulnerability of CNN Classifiers in EEG-Based BCIs." *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 27(5), 814-825.
2. Lawhern, V.J. et al. (2018). "EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain-Computer Interfaces." *Journal of Neural Engineering*, 15(5), 056013.
3. Meng, L. et al. (2024). "Adversarial filtering based evasion and backdoor attacks to EEG-based brain-computer interfaces." *Information Fusion*, 107, 102316.
4. Meng, L., Jiang, X. and Wu, D. (2023). "Adversarial robustness benchmark for EEG-based brain-computer interfaces." *Future Generation Computer Systems*, 143, 231-247.

## Author

Anshuman Deodhar | UMass Amherst | adeodhar@umass.edu
