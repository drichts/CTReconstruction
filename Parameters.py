import numpy as np

# Parameter setting

# Dead pixel mask <asic, row, column>
# Set: dead_pixel_mask[a, r, c] = np.nan
num_asics = 2
dead_pixel_mask = np.ones([num_asics, 24, 36])

ps = 0.330  # Pixel size of the physical detector

# Number of pixels in the reconstructed image in the specified direction
nx = 119  # x-direction
ny = 119  # y-direction
nz = 24  # z-direction (axial direction)

# Physical size of the entire reconstructed image in the specified direction
sx = 30  # x-dir (mm)
sy = 30  # y-dir (mm)
sz = 8  # Axial direction (z) (mm)

# Number of pixels the physical detector along each direction
nu = 36*num_asics  # Longer (horizontal) direction
nv = 24  # Shorter (vertical) direction

# Physical detector size
su = nu*ps  # mm (horizontal dir)
sv = nv*ps  # mm (vertical dir)

DSD = 435  # Distance from the x-ray source to detector (mm)
DSO = 315  # Distance from x-ray source to the axis of rotation (isocenter) (mm)

# Angle settings
direction = 1  # Rotation direction (gantry rotation direction) (1 or -1)
dang = 2  # Angle between captures (degrees)
deg = np.arange(0, 359, dang)  #  List of all capture angles (0-359 by steps of dang)
deg = deg * direction  # Move along angles in the correct rotation direction
num_proj = len(deg)  # Total number of projections

# filter options: 'ram-lak', 'cosine', 'hamming', 'hann'
filter = 'hamming'  # High-pass filter

# Single voxel size (in mm)
dx = sx/nx  # Reconstructed image x-dir voxel
dy = sy/ny  # Reconstructed image y-dir voxel
dz = sz/nz  # Reconstructed image z-dir (axial) voxel

# This is correction for the detector rotation shift (real size, i.e. mm)
off_u, off_v = 0, 0  # Horizontal, vertical

# Geometry calculations for center of each voxel in each direction (in mm measured from the center of the reconstructed
# image or the physical detector)
xs = np.arange(-(nx-1)/2, nx/2, 1) * dx  # Center of voxels in x-dir
ys = np.arange(-(ny-1)/2, ny/2, 1) * dy  # Center of voxels in y-dir
zs = np.arange(-(nz-1)/2, nz/2, 1) * dz  # Center of voxels in the axial direction

us = np.arange(-(nu-1)/2, nu/2, 1) * ps + off_u  # Center of pixels in the physical detector (horizontally)
vs = np.arange(-(nv-1)/2, nv/2, 1) * ps + off_v  # Center of pixels in the physical detector (vertically)

