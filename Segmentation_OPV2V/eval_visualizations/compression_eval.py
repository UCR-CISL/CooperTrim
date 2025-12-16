import matplotlib.pyplot as plt

# Data for all scenarios
scenarios = ['1x', '8x', '32x']

# Compression Dynamic Data
coopertrim_iou_dyn = [54.03, 53.99, 50.32]
cobevt_iou_dyn = [50.23, 50.19, 49.63]
attfuse_iou_dyn = [32.2, 32.21, 32.19]

# Compression Lane Data
coopertrim_iou_lane = [24.45, 24.52, 24.62]
cobevt_iou_lane = [23.79, 23.78, 23.77]
attfuse_iou_lane = [24.14, 24.13, 24.14]

# Compression Road Data
coopertrim_iou_road = [44.38, 44.43, 45.05]
cobevt_iou_road = [45.28, 45.27, 45.27]
attfuse_iou_road = [34.86, 34.86, 34.86]

# Plot 1: Compression Dynamic (Horizontal Bar Chart)
plt.figure(figsize=(5, 5))
bar_height = 0.25
y = range(len(scenarios))
plt.barh([i - bar_height for i in y], coopertrim_iou_dyn, bar_height, label='CooperTrim', color='blue')
plt.barh(y, cobevt_iou_dyn, bar_height, label='CoBEVT', color='lightgreen')
plt.barh([i + bar_height for i in y], attfuse_iou_dyn, bar_height, label='AttFuse', color='red')
plt.ylabel('Compression', fontsize=30)
plt.xlabel('IoU', fontsize=30)
plt.yticks(y, scenarios, fontsize=35)
plt.xticks(fontsize=35)
plt.savefig('compression_dynamic.png', bbox_inches='tight')
plt.close()

# Plot 2: Compression Lane (Horizontal Bar Chart)
plt.figure(figsize=(5, 5))
plt.barh([i - bar_height for i in y], coopertrim_iou_lane, bar_height, label='CooperTrim', color='blue')
plt.barh(y, cobevt_iou_lane, bar_height, label='CoBEVT', color='lightgreen')
plt.barh([i + bar_height for i in y], attfuse_iou_lane, bar_height, label='AttFuse', color='red')
plt.ylabel('Compression', fontsize=30)
plt.xlabel('IoU', fontsize=30)
plt.yticks(y, scenarios, fontsize=35)
plt.xticks(fontsize=35)
plt.savefig('compression_lane.png', bbox_inches='tight')
plt.close()

# Plot 3: Compression Road (Horizontal Bar Chart)
plt.figure(figsize=(5, 5))
plt.barh([i - bar_height for i in y], coopertrim_iou_road, bar_height, label='CooperTrim', color='blue')
plt.barh(y, cobevt_iou_road, bar_height, label='CoBEVT', color='lightgreen')
plt.barh([i + bar_height for i in y], attfuse_iou_road, bar_height, label='AttFuse', color='red')
plt.ylabel('Compression', fontsize=30)
plt.xlabel('IoU', fontsize=30)
plt.yticks(y, scenarios, fontsize=35)
plt.xticks(fontsize=35)
plt.savefig('compression_road.png', bbox_inches='tight')
plt.close()

# Create a separate figure for the legend
fig, ax = plt.subplots(figsize=(5, 5))
ax.barh([0, 0, 0], [0, 0, 0], label='CooperTrim', color='blue')
ax.barh([0, 0, 0], [0, 0, 0], label='CoBEVT', color='lightgreen')
ax.barh([0, 0, 0], [0, 0, 0], label='AttFuse', color='red')
handles, labels = ax.get_legend_handles_labels()
figlegend = plt.figure(figsize=(5, 5))
figlegend.legend(handles, labels, loc='center', fontsize=30)
figlegend.savefig('compression_legend.png', bbox_inches='tight')
plt.close()
