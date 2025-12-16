import matplotlib.pyplot as plt
import numpy as np

# Data for Bandwidth % Used
baselines = ["CoBEVT", "AttFuse", "DiscoNet"]
dynamic_ours = [32.04, 24.76, 10.65]  # Ours percentages for Dynamic
dynamic_baseline = [100, 100, 100]     # Baseline always at 100%
static_lane_ours = [23.77, 17.39, 9.72]
static_lane_baseline = [100, 100, 100]
static_road_ours = [23.77, 17.39, 9.72]
static_road_baseline = [100, 100, 100]

# Data for Accuracy (converted from decimal to percentage)
dynamic_ours_acc = [54.03, 30.90, 30.80]  # Converted to percentage
dynamic_baseline_acc = [50.23, 32.20, 30.03]
static_lane_ours_acc = [24.45, 23.93, 22.05]
static_lane_baseline_acc = [23.79, 24.14, 20.72]
static_road_ours_acc = [44.38, 36.22, 40.02]
static_road_baseline_acc = [45.28, 34.86, 38.43]

# Bar width and positions for grouped bars
bar_width = 0.35
x = np.arange(len(baselines))

# Define consistent colors for "Ours" and "Baseline" across all plots
colors = {
    "Ours": "#ff7f0e",          # Orange for Ours
    "Baseline": "#1f77b4"       # Blue for Baseline
}

# Function to create a dual subplot plot for each scenario
def create_dual_subplot_plot(ours_bw, baseline_bw, ours_acc, baseline_acc, title, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={'hspace': 0.2})
    
    # Top subplot: IOU/Accuracy %
    ax1.bar(x - bar_width/2, baseline_acc, bar_width, color=colors["Baseline"], label="Baseline")
    ax1.bar(x + bar_width/2, ours_acc, bar_width, color=colors["Ours"], label="Ours")
    ax1.set_ylabel('IOU %', fontsize=40)
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax1.tick_params(axis='y', labelsize=40)
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # Hide x-axis for top subplot
    
    # Bottom subplot: Bandwidth %
    ax2.bar(x - bar_width/2, baseline_bw, bar_width, color=colors["Baseline"], label="Baseline")
    ax2.bar(x + bar_width/2, ours_bw, bar_width, color=colors["Ours"], label="Ours")
    ax2.set_ylabel('Bandwidth %', fontsize=40)
    ax2.set_ylim(0, 100)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(baselines, fontsize=40)
    ax2.tick_params(axis='y', labelsize=40)
    
    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.15, hspace=0.3)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Create three separate dual subplot plots for each scenario
# 1. Dynamic
create_dual_subplot_plot(dynamic_ours, dynamic_baseline, dynamic_ours_acc, dynamic_baseline_acc, 
                         'Dynamic Performance', 'ne_dynamic.png')

# 2. Static Lane
create_dual_subplot_plot(static_lane_ours, static_lane_baseline, static_lane_ours_acc, static_lane_baseline_acc, 
                         'Static Lane Performance', 'ne_static.png')

# 3. Static Road
create_dual_subplot_plot(static_road_ours, static_road_baseline, static_road_ours_acc, static_road_baseline_acc, 
                         'Static Road Performance', 'ne_static_road.png')

# Create a separate horizontal legend image with font size 40
fig, ax = plt.subplots(figsize=(10, 2))
# Dummy plots for legend
ax.bar(0, 0, color=colors["Baseline"], label="Baseline")
ax.bar(0, 0, color=colors["Ours"], label="Ours")
# Hide the axes
ax.set_axis_off()
# Add horizontal legend
legend = ax.legend(loc="center", ncol=2, fontsize=40, frameon=True, fancybox=True, shadow=True)
# Save the legend as an image
plt.savefig('ne_legends.png', dpi=300, bbox_inches='tight', transparent=True)
plt.close()
