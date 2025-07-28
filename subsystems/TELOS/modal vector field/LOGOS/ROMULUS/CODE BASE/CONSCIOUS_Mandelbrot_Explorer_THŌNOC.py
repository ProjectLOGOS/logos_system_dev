"""
Mandelbrot Explorer for THŌNOC

Implements visualization and navigation of the Mandelbrot fractal space
that serves as the structural backbone of the THŌNOC system.

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import math

class MandelbrotExplorer:
    """Navigator for the Mandelbrot space representing knowledge in THŌNOC."""
    
    def __init__(self, max_iterations: int = 100, escape_radius: float = 2.0):
        """Initialize the Mandelbrot explorer.
        
        Args:
            max_iterations: Maximum iterations for Mandelbrot calculation
            escape_radius: Escape radius for Mandelbrot calculation
        """
        self.max_iterations = max_iterations
        self.escape_radius = escape_radius
        self.current_center = complex(0, 0)
        self.current_zoom = 1.0
        self.history = []
        self.ontological_markers = {}
        
    def compute_mandelbrot_set(self, 
                               width: int = 800, 
                               height: int = 600, 
                               x_range: Tuple[float, float] = None,
                               y_range: Tuple[float, float] = None) -> np.ndarray:
        """Compute the Mandelbrot set for the current view.
        
        Args:
            width: Width of the output array
            height: Height of the output array
            x_range: Optional x-range override
            y_range: Optional y-range override
            
        Returns:
            Array of iteration counts
        """
        # Calculate view bounds if not provided
        if x_range is None or y_range is None:
            bounds = self.get_view_bounds(width/height)
            x_range = (bounds[0].real, bounds[1].real)
            y_range = (bounds[0].imag, bounds[1].imag)
        
        # Create meshgrid for computation
        x = np.linspace(x_range[0], x_range[1], width)
        y = np.linspace(y_range[0], y_range[1], height)
        c = x.reshape((1, width)) + 1j * y.reshape((height, 1))
        
        # Initialize z and iteration count
        z = np.zeros_like(c)
        iterations = np.zeros(c.shape, dtype=np.int32)
        mask = np.ones(c.shape, dtype=np.bool_)
        
        # Compute iterations
        for i in range(self.max_iterations):
            z[mask] = z[mask]**2 + c[mask]
            new_mask = np.abs(z) < self.escape_radius
            iterations[mask & ~new_mask] = i
            mask = new_mask
            if not np.any(mask):
                break
        
        # Set remaining points to max_iterations
        iterations[mask] = self.max_iterations
        
        return iterations
    
    def zoom_to(self, center: complex, zoom_factor: float) -> None:
        """Zoom to a specific point in the fractal space.
        
        Args:
            center: Center point to zoom to
            zoom_factor: Factor to zoom by
        """
        self.history.append((self.current_center, self.current_zoom))
        self.current_center = center
        self.current_zoom *= zoom_factor
        
    def zoom_out(self) -> bool:
        """Zoom out to previous level.
        
        Returns:
            True if successful, False if history is empty
        """
        if not self.history:
            return False
        
        self.current_center, self.current_zoom = self.history.pop()
        return True
    
    def get_view_bounds(self, aspect_ratio: float = 4/3) -> Tuple[complex, complex]:
        """Get the bounds of the current view.
        
        Args:
            aspect_ratio: Width/height ratio
            
        Returns:
            Tuple of (min_bound, max_bound) as complex numbers
        """
        width = 4.0 / self.current_zoom
        height = width / aspect_ratio
        
        min_bound = complex(
            self.current_center.real - width/2,
            self.current_center.imag - height/2
        )
        max_bound = complex(
            self.current_center.real + width/2,
            self.current_center.imag + height/2
        )
        
        return (min_bound, max_bound)
    
    def add_ontological_marker(self, name: str, 
                              position: complex, 
                              trinity_vector: Tuple[float, float, float]) -> None:
        """Add an ontological marker to the Mandelbrot space.
        
        Args:
            name: Name/identifier for the marker
            position: Complex position in Mandelbrot space
            trinity_vector: (𝔼, 𝔾, 𝕋) dimensions
        """
        self.ontological_markers[name] = {
            "position": position,
            "trinity_vector