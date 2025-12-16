import matplotlib.pyplot as plt

# Data
models = ["CoBeVT", "V2VNet", "Where2Comm", "DiscoNet", "FCooper", "OpV2V", "V2X-ViT" ]
network_usage = [41, 800, 120.6, 83.8, 104, 5324.8, 27]  # Bandwidth in Mbps
ap_iou_0_7 = [84, 91.4, 59, 83.1, 79, 81.5, 88.2]
ap_iou_0_5 = [80, 82.2, 47, 83.6, 78.8, 81, 71.2]  # Values for AP@IOU 0.5

# Define consistent colors and marker shapes for each network
model_styles = {
    "CoBeVT": {"color": "blue", "marker": "o"},
    "V2VNet": {"color": "green", "marker": "s"},
    "Where2Comm": {"color": "orange", "marker": "^"},
    "DiscoNet": {"color": "red", "marker": "d"},
    "FCooper": {"color": "purple", "marker": "x"},
    "OpV2V": {"color": "cyan", "marker": "v"},
    "V2X-ViT": {"color": "magenta", "marker": "p"}
}

# Plot for AP@IOU 0.5
plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(network_usage[i], ap_iou_0_5[i], color=model_styles[model]["color"],
                marker=model_styles[model]["marker"], label=f"{model}")
    plt.text(network_usage[i] + 10, ap_iou_0_5[i], f"{model} ({network_usage[i]} Mbps)", fontsize=9, ha='left', va='center')
plt.xlabel("Network Bandwidth (Mbps)", fontsize=12)
plt.ylabel("AP@IOU", fontsize=12)
plt.title("AP@IOU 0.5 vs Network Bandwidth", fontsize=14)
# plt.legend(loc="upper left", fontsize=10, frameon=False)
plt.grid(False)
plt.xlim(0, max(network_usage) + 100)
plt.ylim(min(ap_iou_0_5) - 5, max(ap_iou_0_5) + 5)
plt.savefig("ap_iou_0_5_vs_network_bandwidth_custom.png", bbox_inches='tight')
plt.show()

# Plot for AP@IOU 0.7
plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(network_usage[i], ap_iou_0_7[i], color=model_styles[model]["color"],
                marker=model_styles[model]["marker"], label=f"{model}")
    plt.text(network_usage[i] + 10, ap_iou_0_7[i], f"{model} ({network_usage[i]} Mbps)", fontsize=9, ha='left', va='center')
plt.xlabel("Network Bandwidth (Mbps)", fontsize=12)
plt.ylabel("AP@IOU", fontsize=12)
plt.title("AP@IOU 0.7 vs Network Bandwidth", fontsize=14)
# plt.legend(loc="upper left", fontsize=10, frameon=False)
plt.grid(False)
plt.xlim(0, max(network_usage) + 100)
plt.ylim(min(ap_iou_0_7) - 5, max(ap_iou_0_7) + 5)
plt.savefig("ap_iou_0_7_vs_network_bandwidth_custom.png", bbox_inches='tight')
plt.show()
print("done")
