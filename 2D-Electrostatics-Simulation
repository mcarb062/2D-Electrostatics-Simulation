import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def two_dimension_jacobi(schematic, boundary_location, error_threshold, middle_matrix):
    height, width = schematic.shape
    input_matrix = schematic.copy()
    output_matrix = schematic.copy()
    second_matrix = schematic.copy()

    iterations = 0
    delta_v = 0

    while True:
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                if (i, j) not in boundary_location:
                    output_matrix[i, j] = (
                        input_matrix[i - 1, j]
                        + input_matrix[i - 1, j - 1]
                        + input_matrix[i - 1, j + 1]
                        + input_matrix[i + 1, j]
                        + input_matrix[i + 1, j - 1]
                        + input_matrix[i + 1, j + 1]
                        + input_matrix[i, j - 1]
                        + input_matrix[i, j + 1]
                    ) / 8
                    delta_v += abs(input_matrix[i, j] - output_matrix[i, j])

        iterations += 1
        if iterations == middle_matrix:
            second_matrix = output_matrix.copy()

        if delta_v < error_threshold:
            break
        else:
            delta_v = 0
            input_matrix = output_matrix.copy()

    return output_matrix, second_matrix, iterations, delta_v

def plot_results(box, output_matrix, iterations):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(box, cmap="bone")
    axs[1].imshow(output_matrix[1], cmap="bone")
    axs[2].imshow(output_matrix[0], cmap="bone")
    titles = ['Voltage after 0 Iterations', f'Voltage after {iterations} iterations', f'Voltage after {iterations} Iterations']
    for ax, title in zip(axs, titles):
        ax.set_title(title)
    plt.tight_layout()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection="3d")
    x = np.linspace(0, width, bins)
    y = np.linspace(0, height, bins)
    x, y = np.meshgrid(x, y)
    ax1.plot_surface(x, y, output_matrix[0], cmap="bone")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("Voltage")

    plt.show()

# Constants
A, B, C, D, E = 1, 2, 0.5, 3, 2.5
deciVolt, milliVolt = 0.1, 0.001
width, height = 1, 1
bins = 20
dx = width / bins
minima, maxima = -1.0, 1.0
box = np.zeros((bins, bins), dtype=float)
boundary_location = []

for i in range(bins):
    box[i][0] = A - C * (dx * (i + 1)) ** 2
    box[i][bins - 1] = D - E * (dx * (i + 1)) ** 2
    box[0][i] = A + B * (dx * (i + 1))
    box[bins - 1][i] = A / 2
    boundary_location.extend([(i, 0), (i, width - dx), (0, i), (width - dx, i)])

output_matrix, second_matrix, iterations, delta_v = two_dimension_jacobi(
    box, boundary_location, milliVolt, middle_matrix=10
)

plot_results(box, [output_matrix, second_matrix], iterations)
