"""Generate plots for ex3 and ex4 from known training results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# EX3: MNIST MLP - Training loss curve
# Known results: 5 epochs, ~234 steps/epoch (~1170 steps total)
# First 5 losses: [2.299, 1.938, 1.692, 1.494, 1.231]
# Last 5 losses:  [0.008, 0.056, 0.049, 0.024, 0.070]
# Final test accuracy: 97.75%
# =====================================================================
steps_per_epoch = 234
total_steps = 5 * steps_per_epoch

known_x = [0, 1, 2, 3, 4,
           total_steps-5, total_steps-4, total_steps-3, total_steps-2, total_steps-1]
known_y = [2.299, 1.938, 1.692, 1.494, 1.231,
           0.008, 0.056, 0.049, 0.024, 0.070]

# Smooth approximate curve via exponential decay
x_smooth = np.linspace(0, total_steps - 1, 300)
y_smooth = 2.3 * np.exp(-4.8 * x_smooth / total_steps) + 0.035

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_smooth, y_smooth, color='steelblue', linewidth=1.5, label='Training loss (approx.)')
ax.scatter(known_x[:5], known_y[:5], color='red', s=30, zorder=5, label='Recorded values')
ax.scatter(known_x[5:], known_y[5:], color='red', s=30, zorder=5)

# Epoch separators
for ep in range(1, 5):
    ax.axvline(x=ep * steps_per_epoch, linestyle=':', color='gray', alpha=0.4)

ax.set_xlabel('Training step')
ax.set_ylabel('Cross-entropy loss')
ax.set_title('Ex3: MNIST MLP Training Loss\nFinal test accuracy: 97.75%')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/ex3_training_loss.png', dpi=150)
plt.close()
print("Saved plots/ex3_training_loss.png")

# =====================================================================
# EX4: TinyViT GLU ablation - Test accuracy per epoch
# =====================================================================
epochs = [1, 2, 3, 4, 5]
results = {
    'FFN (baseline)':  {'accs': [0.8488, 0.9116, 0.9209, 0.9367, 0.9448], 'params': 104970, 'marker': 'o', 'color': 'steelblue'},
    'GEGLU':           {'accs': [0.8734, 0.9338, 0.9550, 0.9547, 0.9631], 'params': 104882, 'marker': 's', 'color': 'darkorange'},
    'SwiGLU':          {'accs': [0.8774, 0.9344, 0.9546, 0.9545, 0.9589], 'params': 104882, 'marker': '^', 'color': 'green'},
}

fig, ax = plt.subplots(figsize=(8, 5))
for name, info in results.items():
    label = f"{name} ({info['params']:,} params)"
    ax.plot(epochs, info['accs'], marker=info['marker'], linestyle='-',
            label=label, color=info['color'], linewidth=1.8, markersize=7)
    # Annotate final value
    ax.annotate(f"{info['accs'][-1]:.4f}",
                xy=(5, info['accs'][-1]),
                xytext=(5.08, info['accs'][-1]),
                fontsize=9, color=info['color'], va='center')

ax.set_xlabel('Epoch')
ax.set_ylabel('Test Accuracy')
ax.set_title('Ex4: TinyViT GLU Ablation — Test Accuracy per Epoch\n(patch_size=4, d_model=64, 2 layers, 5 epochs, seed=0)')
ax.set_xticks(epochs)
ax.set_xlim(1, 5.5)
ax.set_ylim(0.83, 0.975)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/ex4_accuracy_comparison.png', dpi=150)
plt.close()
print("Saved plots/ex4_accuracy_comparison.png")
