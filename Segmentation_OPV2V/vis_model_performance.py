from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np

# Function to apply a moving average for smoothing
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Specify the log directory
# log_dir = 'C:/Shilpa/UCR_PhD/research/logs_autorl/100_perc_request/events.out.tfevents.1744574974.dune1.cris.local'
# log_dir = 'C:/Shilpa/UCR_PhD/research/logs_autorl/cobevt100_perc_request/events.out.tfevents.1744586313.dune1.cris.local'
log_dir = 'C:/Shilpa/UCR_PhD/research/logs_autorl/whole100_perc_request/events.out.tfevents.1744585444.dune1.cris.local'

# Create an EventAccumulator object
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()  # Load the event data

# Extract data for a specific scalar
scalar_data = ea.Scalars('Dynamic_Iou')
# scalar_data = ea.Scalars('Dynamic_loss')

# Separate the data into x (steps) and y (values)
steps = [x.step for x in scalar_data]
values = [x.value for x in scalar_data]

# Apply moving average to smoothen the values
window_size = 1 #100  # Adjust the window size for more or less smoothing
smoothed_values = moving_average(values, window_size)

# Adjust steps to match the smoothed values
smoothed_steps = steps[:len(smoothed_values)]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(smoothed_steps, smoothed_values, label='IOU')
# plt.plot(smoothed_steps, smoothed_values, label='Validation Loss')
plt.xlabel('Steps')
plt.ylabel('Values')
plt.title('Metric over Steps')
plt.legend()
plt.show()
