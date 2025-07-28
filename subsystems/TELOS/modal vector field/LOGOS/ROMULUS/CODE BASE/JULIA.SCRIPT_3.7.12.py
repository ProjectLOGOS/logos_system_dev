import json
import numpy as np
import matplotlib.pyplot as plt
import os
from numba import jit, prange

# Ensure OUTPUT directory exists
OUTPUT_DIR = "OUTPUT"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load the property mapping from a JSON file
def load_properties(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Load properties at runtime
property_mapping = load_properties('JULIA.DICTIONARY_3.7.12.json')

# Supported color maps
SUPPORTED_COLOR_MAPS = ["magma", "plasma", "inferno", "viridis", "cividis", "twilight", "coolwarm"]

# Optimized function to compute the Julia set using Numba-compatible grid computation
@jit(nopython=True, parallel=True)
def compute_julia_set(width, height, zoom, c_value, max_iter, escape_radius):
    x = np.linspace(-1.5 / zoom, 1.5 / zoom, width)
    y = np.linspace(-1.5 / zoom, 1.5 / zoom, height)
    
    julia = np.zeros((height, width), dtype=np.int32)
    
    # Loop over each pixel directly without using meshgrid
    for i in prange(height):
        for j in range(width):
            zx, zy = x[j], y[i]
            z = complex(zx, zy)
            count = 0
            while abs(z) < escape_radius and count < max_iter:
                z = z**2 + c_value
                count += 1
            julia[i, j] = count
    return julia

# Function to display the Julia set with a color map
def plot_julia_set(julia, property_name, save_path=None, color_map='magma'):
    if color_map not in SUPPORTED_COLOR_MAPS:
        print(f"Warning: Unsupported color map '{color_map}'. Defaulting to 'magma'.")
        color_map = 'magma'

    plt.figure(figsize=(10, 10))
    plt.imshow(julia, cmap=color_map, extent=(-1.5, 1.5, -1.5, 1.5))
    plt.axis('off')
    plt.title(property_name, fontsize=20, pad=20, color="black", backgroundcolor="white")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Julia set saved as {save_path}")
    else:
        plt.show()

# Generate a Julia set for a specific property
def generate_julia(property_name, color_map='magma'):
    if property_name not in property_mapping:
        print(f"Error: Property '{property_name}' not found in the mapping.")
        return

    property_data = property_mapping[property_name]
    print(f"Generating Julia set for {property_name}...")

    # Use the unified c_value field
    real = property_data["c_value"]["real"]
    imaginary = property_data["c_value"]["imaginary"]
    c_value = complex(real, imaginary)

    width, height = 2000, 2000  # High-quality resolution
    zoom = 1.0
    max_iter = 550  # High-quality iteration count
    escape_radius = 2.0

    # Compute the Julia set
    julia_set = compute_julia_set(width, height, zoom, c_value, max_iter, escape_radius)

    # Save the output
    save_path = os.path.join(OUTPUT_DIR, f"{property_name}3.7.12.png")
    plot_julia_set(julia_set, property_name, save_path=save_path, color_map=color_map)

# Main function for command-line execution
def main():
    # Generate all Julia sets with default color map
    print("Generating all Julia sets with default color map 'magma'...")
    for prop in property_mapping.keys():
        generate_julia(prop)

if __name__ == "__main__":
    main()
