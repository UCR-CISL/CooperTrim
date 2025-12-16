# import matplotlib.pyplot as plt
# import numpy as np

# # Data from the table (simplified for plotting)
# baselines = [
#     "Ours-CoBEVT", "STAMP", "UniSense", "SwissCheese", 
#     "Where2Comm", "CoBEVT", "AttFuse", "V2X-ViT", 
#     "DiscoNet", "V2VNet", "FCooper"
# ]
# bandwidths = [
#     11.165,  # ours-CoBEVT: average of 12.82 and 9.51
#     320,     # STAMP: average of 320 and 80
#     75,      # UniSense
#     81.92,   # SwissCheese
#     40.3,    # Where2Comm: assuming N=1 for simplicity
#     40,      # CoBEVT
#     5324.8,  # AttFuse
#     27,      # V2X-ViT
#     83.8,    # DiscoNet
#     800,     # V2VNet
#     104      # FCooper: average of 5872 and 104
# ]
# selection_types = [
#     "FS", "C", "FS", "FS", "FS+AS", "C", "C", "C", "C", "C", "C"
# ]

# # Compute logarithmic bandwidths
# log_bandwidths = np.log10(bandwidths)

# # Define colors and markers for each selection type
# styles = {
#     "FS": {"color": "blue", "marker": "^", "label": "FS"},
#     "C": {"color": "red", "marker": "o", "label": "C"},
#     "FS+AS": {"color": "cyan", "marker": "s", "label": "FS+AS"},
# }

# # Plotting
# plt.figure(figsize=(12, 11))
# seen_labels = set()  # To avoid duplicate labels in legend
# for i in range(len(baselines)):
#     sel_type = selection_types[i]
#     style = styles[sel_type if sel_type != "FS" or baselines[i] != "Ours-CoBEVT" else "FS"]
#     label = style["label"] if style["label"] not in seen_labels else None
#     if label:
#         seen_labels.add(style["label"])
#     plt.scatter(i, bandwidths[i], c=style["color"], marker=style["marker"], s=100, label=label)
#     # Special handling for Ours-CoBEVT to distinguish it within FS
#     if baselines[i] == "Ours-CoBEVT":
#         plt.scatter(i, bandwidths[i], c="magenta", marker="^", s=150, label="Ours (FS)")

# # Customize the plot with log scale on y-axis and larger font size
# plt.xticks(range(len(baselines)), baselines, rotation=45, ha="right", fontsize=30)
# plt.yticks(fontsize=30)
# # plt.xlabel("Baseline", fontsize=30)
# plt.ylabel("Bandwidth (Mbps) Log Scale", fontsize=30)
# plt.yscale('log')  # Set y-axis to logarithmic scale
# # plt.legend(loc="upper left", fontsize=30)
# plt.grid(True, linestyle="--", alpha=0.7, which='both')  # Show grid for both major and minor ticks
# # Place legend horizontally below the plot
# plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=4, fontsize=30)
# plt.tight_layout()

# # Save the plot instead of showing it
# plt.savefig("bandwidth_comparison.png", dpi=300, bbox_inches="tight")
# plt.close()  # Close the figure to free memory

import matplotlib.pyplot as plt
import numpy as np

# Data from the table (simplified for plotting)
baselines = [
    "CooperTrim", "STAMP", "UniSense", "SwissCheese", 
    "Where2Comm", "CoBEVT", "AttFuse", "V2X-ViT", 
    "DiscoNet", "V2VNet", "FCooper"
]
bandwidths = [
    11.165,  # ours-CoBEVT: average of 12.82 and 9.51
    320,     # STAMP: average of 320 and 80
    75,      # UniSense
    81.92,   # SwissCheese
    40.3,    # Where2Comm: assuming N=1 for simplicity
    40,      # CoBEVT
    5324.8,  # AttFuse
    27,      # V2X-ViT
    83.8,    # DiscoNet
    800,     # V2VNet
    104      # FCooper: average of 5872 and 104
]
selection_types = [
    "FS", "C", "FS", "FS", "FS+AS", "C", "C", "C", "C", "C", "C"
]

# Define colors for each selection type
styles = {
    "FS": {"color": "gray", "label": "FS"},
    "C": {"color": "green", "label": "C"},
    "FS+AS": {"color": "orange", "label": "FS+AS"},
    "CooperTrim": {"color": "blue", "label": "CooperTrim (FS)"}
}

# Plotting
plt.figure(figsize=(13, 11))
x = np.arange(len(baselines))
bar_width = 0.8  # Slightly less than 1 to create small whitespace between bars

# Track which labels have been added to avoid duplicates
seen_labels = set()

# Create bars with colors based on selection type
for i in range(len(baselines)):
    sel_type = selection_types[i]
    if baselines[i] == "CooperTrim":
        color = styles["CooperTrim"]["color"]
        label = styles["CooperTrim"]["label"] if styles["CooperTrim"]["label"] not in seen_labels else None
        if label:
            seen_labels.add(label)
        plt.bar(x[i], bandwidths[i], bar_width, color=color, label=label)
    else:
        color = styles[sel_type]["color"]
        label = styles[sel_type]["label"] if styles[sel_type]["label"] not in seen_labels else None
        if label:
            seen_labels.add(label)
        plt.bar(x[i], bandwidths[i], bar_width, color=color, label=label)

# Customize the plot with log scale on y-axis and larger font size
plt.xticks(x, baselines, rotation=45, ha="right", fontsize=30)
plt.yticks(fontsize=30)
plt.ylabel("Bandwidth (Mbps) Log Scale", fontsize=30)
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.grid(True, linestyle="--", alpha=0.7, which='both')  # Show grid for both major and minor ticks
# Place legend horizontally below the plot
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=4, fontsize=30)
plt.tight_layout()

# Save the plot instead of showing it
plt.savefig("bandwidth_comparison.png", dpi=300, bbox_inches="tight")
plt.close()  # Close the figure to free memory
