import matplotlib.pyplot as plt
import numpy as np

# Data for the bar chart
categories = ['Dynamic', 'Static\nRoad', 'Static\nLane']
latency_0ms = [54.03, 44.38, 24.45]    # IOU values for 0ms
latency_50ms = [54.03, 44.38, 24.45]   # IOU values for 50ms
latency_100ms = [53.88, 44.38, 24.44]  # IOU values for 100ms
latency_200ms = [53.5, 44.29, 24.38]   # IOU values for 200ms

# Set the width of the bars
bar_width = 0.2

# Set position of bar on X axis
r1 = np.arange(len(categories))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Create the bar chart
plt.bar(r1, latency_0ms, color='skyblue', width=bar_width, label='0ms')
plt.bar(r2, latency_50ms, color='lightgreen', width=bar_width, label='50ms')
plt.bar(r3, latency_100ms, color='salmon', width=bar_width, label='100ms')
plt.bar(r4, latency_200ms, color='magenta', width=bar_width, label='200ms')

# Add labels and title with font size 30
plt.ylabel('IOU (%)', fontsize=30)

# Set tick labels font size
plt.xticks([r + bar_width * 1.5 for r in range(len(categories))], categories, fontsize=20, rotation=0)
plt.yticks([10, 30, 50], fontsize=30)

# Add legend at the top center with font size 20
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.55), ncol=2, fontsize=20)

# Adjust layout to prevent label clipping
plt.tight_layout()

# Save the plot
plt.savefig('latency_plot.png')

# Display the plot
plt.show()
