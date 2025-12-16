import matplotlib.pyplot as plt
import numpy as np

# Data for the bar chart
categories = ['Dynamic', 'Static\nRoad', 'Static\nLane']
error_0cm = [54.03, 44.38, 24.45]  # IOU values for 0cm
error_20cm = [53.69, 44.16, 24.19]  # IOU values for 20cm
error_1m = [48.54, 42.66, 21.85]    # IOU values for 1m

# Set the width of the bars
bar_width = 0.25

# Set position of bar on X axis
r1 = np.arange(len(categories))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Create the bar chart
plt.bar(r1, error_0cm, color='skyblue', width=bar_width, label='0cm')
plt.bar(r2, error_20cm, color='lightgreen', width=bar_width, label='±20cm')
plt.bar(r3, error_1m, color='salmon', width=bar_width, label='±1m')

# Add labels and title with font size 30
# plt.xlabel('Category', fontsize=30)
plt.ylabel('IOU (%)', fontsize=30)
# plt.title('IOU vs Category by Localization Error', fontsize=30)

# Set tick labels font size to 30
plt.xticks([r + bar_width for r in range(len(categories))], categories, fontsize=20, rotation=0)
plt.yticks([10, 30, 50], fontsize=30)

# Add legend at the top center with font size 30
plt.legend(loc='upper center', bbox_to_anchor=(0.35, 1.55), ncol=3, fontsize=20)


# Adjust layout to prevent label clipping
plt.tight_layout()

# Save the plot
plt.savefig('localization_error_plot.png')

# Display the plot
plt.show()
