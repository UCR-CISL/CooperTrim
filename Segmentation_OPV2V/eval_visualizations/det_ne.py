import matplotlib.pyplot as plt
import numpy as np

# Data for 3D Detection Results
datasets = ["OPV2V", "V2V4Real"]
baseline_ap_05 = [0.88, 0.61]  # Baseline AP@IoU 0.5
coopertrim_ap_05 = [0.80, 0.53]  # CooperTrim AP@IoU 0.5
baseline_ap_07 = [0.78, 0.42]  # Baseline AP@IoU 0.7
coopertrim_ap_07 = [0.69, 0.31]  # CooperTrim AP@IoU 0.7
bandwidth_use = [56.84, 26.83]  # Bandwidth use % vs Baseline
baseline_bandwidth = [100, 100]  # Baseline bandwidth always at 100%

# Convert AP values to percentages for display
baseline_ap_05 = [x * 100 for x in baseline_ap_05]
coopertrim_ap_05 = [x * 100 for x in coopertrim_ap_05]
baseline_ap_07 = [x * 100 for x in baseline_ap_07]
coopertrim_ap_07 = [x * 100 for x in coopertrim_ap_07]

# Bar width and positions for grouped bars
bar_width = 0.35
x = np.arange(len(datasets))

# Define consistent colors for "Baseline" and "CooperTrim"
colors = {
    "Baseline": "#1f77b4",  # Blue for Baseline
    "CooperTrim": "#ff7f0e"  # Orange for CooperTrim
}

# Create a figure with three subplots (AP@0.5, AP@0.7, Bandwidth)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18), sharex=True, gridspec_kw={'hspace': 0.2})

# Subplot 1: AP@IoU 0.5
ax1.bar(x - bar_width/2, baseline_ap_05, bar_width, color=colors["Baseline"], label="Baseline")
ax1.bar(x + bar_width/2, coopertrim_ap_05, bar_width, color=colors["CooperTrim"], label="CooperTrim")
ax1.set_ylabel('AP@IoU\n 0.5 (%)', fontsize=60)  # Increased by 3
ax1.set_ylim(0, 100)
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
ax1.tick_params(axis='y', labelsize=45)  # Increased by 3
ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # Hide x-axis for top subplot

# Subplot 2: AP@IoU 0.7
ax2.bar(x - bar_width/2, baseline_ap_07, bar_width, color=colors["Baseline"], label="Baseline")
ax2.bar(x + bar_width/2, coopertrim_ap_07, bar_width, color=colors["CooperTrim"], label="CooperTrim")
ax2.set_ylabel('AP@IoU\n 0.7 (%)', fontsize=60)  # Increased by 3
ax2.set_ylim(0, 100)
ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
ax2.tick_params(axis='y', labelsize=45)  # Increased by 3
ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # Hide x-axis for middle subplot

# Subplot 3: Bandwidth Usage
ax3.bar(x - bar_width/2, baseline_bandwidth, bar_width, color=colors["Baseline"], label="Baseline")
ax3.bar(x + bar_width/2, bandwidth_use, bar_width, color=colors["CooperTrim"], label="CooperTrim")
ax3.set_ylabel('Bandwidth\n Use (%)', fontsize=60)  # Increased by 3
ax3.set_ylim(0, 100)
ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
ax3.set_xticks(x)
ax3.set_xticklabels(datasets, fontsize=45)  # Increased by 3
ax3.tick_params(axis='y', labelsize=40)  # Increased by 3

# Adjust layout
plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.15, hspace=0.3)
plt.savefig('3d_detection_results.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a separate horizontal legend image with font size increased by 3
fig, ax = plt.subplots(figsize=(10, 2))
# Dummy plots for legend
ax.bar(0, 0, color=colors["Baseline"], label="Baseline")
ax.bar(0, 0, color=colors["CooperTrim"], label="CooperTrim")
# Hide the axes
ax.set_axis_off()
# Add horizontal legend
legend = ax.legend(loc="center", ncol=2, fontsize=33, frameon=True, fancybox=True, shadow=True)  # Increased by 3
# Save the legend as an image
plt.savefig('3d_detection_legends.png', dpi=300, bbox_inches='tight', transparent=True)
plt.close()
