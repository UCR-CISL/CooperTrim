import matplotlib.pyplot as plt
import numpy as np
import textwrap

# Data for the plots
methods = ["EG+CP+FT (Ours)", "EG+SD+FT", "Curriculum+CP+FT", "CP+FT"]

# Accuracy data (converted to percentage for better readability)
dynamic_acc = [54.03, 52.55, 52.47, 51.57]  # Dynamic * 100
static_lane_acc = [24.45, 23.69, 28.27, 23.05]  # Static Lane * 100
static_road_acc = [44.38, 38.45, 45.44, 36.92]  # Static Road * 100

# Bandwidth data (already in percentage)
dynamic_request = [32.04, 35.64, 49.25, 30.75]  # Dynamic Request (%)
static_request = [23.77, 11.03, 51.63, 1.12]    # Static Request (%)

# Define consistent colors for each method (same across all plots)
colors = {
    "EG+CP+FT (Ours)": "#1f77b4",      # Blue
    "EG+SD+FT": "#ff7f0e",             # Orange
    "Curriculum+CP+FT": "#2ca02c",      # Green
    "CP+FT": "#d62728"                 # Red
}

# Function to create a figure with two subplots (IoU/Accuracy on top, Bandwidth on bottom)
def create_dual_subplot_plot(acc_data, bw_data, title, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})
    
    # Top subplot: IoU/Accuracy
    x = np.arange(len(methods))
    # Set bar width to 1.0 to eliminate spacing between bars
    bar_width = 1.0
    ax1.bar(x, acc_data, bar_width, color=[colors[method] for method in methods], align='edge')
    ax1.set_ylabel('IOU %', fontsize=40)
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax1.tick_params(axis='y', labelsize=40)
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # Completely hide x-axis for top subplot
    
    # Bottom subplot: Bandwidth
    ax2.bar(x, bw_data, bar_width, color=[colors[method] for method in methods], align='edge')
    ax2.set_ylabel('BW %', fontsize=40)
    ax2.set_ylim(0, 100)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax2.set_xticklabels([])
    # ax2.set_xticks(x + bar_width/2)  # Center the ticks between bars
    # Wrap x-axis labels to fit within a certain width (e.g., 10 characters per line)
    wrapped_labels = [textwrap.fill(label, width=13) for label in methods]
    # ax2.set_xticklabels(wrapped_labels, fontsize=40, rotation=45, ha='right')  # Tilt labels at 45 degrees
    ax2.tick_params(axis='y', labelsize=40)
    
    # Add legend at the bottom
    # fig.legend(methods, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=30, frameon=True, fancybox=True, shadow=True)
    # Adjust layout to eliminate any padding or spacing
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.2, hspace=0, wspace=0)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Create three separate plots
# 1. Dynamic Plot
create_dual_subplot_plot(dynamic_acc, dynamic_request, 'Dynamic Performance', 'ablation_dynamic_plot.png')

# 2. Static Lane Plot
create_dual_subplot_plot(static_lane_acc, static_request, 'Static Lane Performance', 'ablation_static_lane_plot.png')

# 3. Static Road Plot
create_dual_subplot_plot(static_road_acc, static_request, 'Static Road Performance', 'ablation_static_road_plot.png')

# Create a figure just for the legend
fig, ax = plt.subplots(figsize=(10, 2))  # Adjust size as needed

# Create dummy plots for legend (we won't display them, just use for legend)
for method in methods:
    ax.bar(0, 0, color=colors[method], label=method)

# Hide the axes and any content, we only want the legend
ax.set_axis_off()

# Add legend horizontally
legend = ax.legend(methods, loc="center", ncol=len(methods), fontsize=12, frameon=True, fancybox=True, shadow=True)

# Save the legend as an image
plt.savefig('ablation_legend.png', dpi=300, bbox_inches='tight', transparent=True)
plt.close()
