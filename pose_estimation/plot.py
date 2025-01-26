import matplotlib.pyplot as plt

# Open the file in read mode
with open("to_plot.txt", "r") as file:
    time_steps = []
    y1_values = []
    y2_values = []
    speed_values = []

    # Iterate through each line in the file
    for line in file:
        # Strip whitespace and split the line by spaces
        elements = line.strip().split(' ')
        
        # Extract the four elements as floats
        if len(elements) == 4:  # Ensure there are exactly 4 elements in the line
            time = float(elements[0])
            y1 = float(elements[1])
            y2 = float(elements[2])
            speed = float(elements[3])

            # Print the values (optional)
            print(f"Time: {time}, Y1: {y1}, Y2: {y2}, Speed: {speed}")
            
            # Append data to the lists
            time_steps.append(time)
            y1_values.append(y1)  # Store y1 values for plotting
            y2_values.append(y2)  # Store y2 values for plotting
            speed_values.append(speed)  # Store speed values for plotting
    print(speed_values)
    print("size: ", len(speed_values))

# Create the figure and axes for subplots

for i in range(400, len(speed_values), 400):
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))

    # Plot on the first subplot (top)
    ax[0].plot(time_steps[i-400:i], y1_values[i-400:i], label="Y1 Value", color='blue')
    ax[0].plot(time_steps[i-400:i], y2_values[i-400:i], label="Y2 Value", color='green')
    ax[0].set_title("Y1 and Y2 Values over Time")
    ax[0].set_xlabel("Time Step")
    ax[0].set_ylabel("Value")
    ax[0].legend()

    # Plot on the second subplot (bottom)
    ax[1].plot(time_steps[i-400:i], speed_values[i-400:i], label="Speed", color='orange')
    ax[1].set_title("Speed over Time")
    ax[1].set_xlabel("Time Step")
    ax[1].set_ylabel("Speed")
    ax[1].legend()

# Adjust layout to avoid overlapping
plt.tight_layout()

# Display the plots
plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # Read data from text file
# timestamps, left_angles, right_angles = [], [], []
# with open("rachel_new_data.txt", "r") as file:
#     for line in file:
#         t, left, right = map(float, line.split())
#         timestamps.append(t)
#         left_angles.append(left)
#         right_angles.append(right)

# # Plot the data
# plt.figure(figsize=(10, 6))
# plt.plot(timestamps, left_angles, label="Left Angle")
# plt.plot(timestamps, right_angles, label="Right Angle")
# plt.xlabel("Time (s)")
# plt.ylabel("Angle (degrees)")
# plt.title("Angle Between Hip, Knee, and Ankle Over Time")
# plt.legend()
# plt.show()
