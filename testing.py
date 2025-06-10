
import matplotlib.pyplot as plt

# Sample data
x = [0, 1, 2, 3, 4]
y = [0, 1, 4, 9, 16]       # Left y-axis
z = [100, 80, 60, 40, 20]  # Right y-axis ("z")

# Create figure and axis
fig, ax1 = plt.subplots()

# Plot on the left y-axis
ax1.plot(x, y, 'b-', label='Y data')
ax1.set_xlabel('X Axis')
ax1.set_ylabel('Y Axis', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a twin y-axis on the right for the "Z" axis
ax2 = ax1.twinx()
ax2.plot(x, z, 'r--', label='Z data')
ax2.set_ylabel('Z Axis', color='r')
ax2.tick_params(axis='y', labelcolor='r')

ax1.legend()

l = ax1.get_legend()
print(type(l))
quit()

# Optional: combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.title('Dual Y-Axis Plot (Y and "Z")')
plt.show()

