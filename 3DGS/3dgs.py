import mlx
import mlx.nn as nn
import mlx.core as mx
import numpy as np

# I should probably transfer the comments in ResNet into the README.md
"""
Implementation details:
    * Started optimization with 4 times smaller image resolution and upsampled twice after 250 and 500 iterations
    * If there's a missing regions when capturing cornes or 'inside out' captures the spherical harmonic values will be wrong; solved by optimizing only zero-order component and adding one band of the SH after every 1K iterations until all 4 are represented (?)
        * Only base color is considered at first before adding more fancy lighting at different angles 
        * SH bands represent levels of detail; represents functions on a sphere. Makes sense in how they'd represent lighting variations.
    * Initialized from SFM point cloud
    * Densification: Clone and split
"""

### Gonna implement SfM first brb
