import matplotlib.pyplot as plt

epsilons = [0.01, 0.05, 0.10, 0.20, 0.30]

# 3-subject scope results (from the filter attack run on Day 2)
baseline_clean = 57.51
defended_clean = 57.80
baseline_fgsm = [52.02, 35.26, 16.76, 2.02, 0.00]
defended_fgsm = [56.36, 47.98, 37.86, 20.52, 10.40]
baseline_filter = [55.20, 49.42, 40.75, 26.59, 33.24]
defended_filter = [56.94, 54.62, 45.38, 37.28, 34.39]

# WIDER figure to fit titles cleanly
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left panel: baseline
axes[0].axhline(y=baseline_clean, color='g', linestyle=':', label=f'Clean ({baseline_clean:.1f}%)', linewidth=1.5)
axes[0].plot(epsilons, baseline_fgsm, 'r-o', label='FGSM (per-trial)', linewidth=2)
axes[0].plot(epsilons, baseline_filter, 'm-s', label='Universal filter', linewidth=2)
axes[0].axhline(y=25, color='gray', linestyle=':', label='Random chance (25%)', linewidth=1.5)
axes[0].set_xlabel('Epsilon (relative to signal std)')
axes[0].set_ylabel('Classification Accuracy (%)')
axes[0].set_title('Baseline: FGSM vs Universal Filter (3-subj)', fontsize=12)
axes[0].legend(loc='upper right', fontsize=9)
axes[0].grid(True, alpha=0.3)

# Right panel: defended
axes[1].axhline(y=defended_clean, color='g', linestyle=':', label=f'Clean ({defended_clean:.1f}%)', linewidth=1.5)
axes[1].plot(epsilons, defended_fgsm, 'b-o', label='FGSM (per-trial)', linewidth=2)
axes[1].plot(epsilons, defended_filter, 'm-s', label='Universal filter', linewidth=2)
axes[1].axhline(y=25, color='gray', linestyle=':', label='Random chance (25%)', linewidth=1.5)
axes[1].set_xlabel('Epsilon (relative to signal std)')
axes[1].set_ylabel('Classification Accuracy (%)')
axes[1].set_title('Defended: FGSM vs Universal Filter (3-subj)', fontsize=12)
axes[1].legend(loc='lower left', fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('filter_attack_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved filter_attack_results.png with fixed layout")