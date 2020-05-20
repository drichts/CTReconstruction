import numpy as np

# Parameter setting

ps = 0.330  # Pixel size of the phyiscal detector

# Number of pixels in the reconstructed image in the specified direction
nx = 120  # x-direction
ny = 120  # y-direction
nz = 24  # z-direction (axial direction)

# Physical size of the entire reconstructed image in the specified direction
sx = 30  # x-dir (mm)
sy = 30  # y-dir (mm)
sz = 8  # Axial direction (z) (mm)

# Number of pixels the physical detector along each direction
nu = 72  # Longer (horizontal) direction
nv = 24  # Shorter (vertical) direction

# Physical detector size
su = nu*ps  # mm (horizontal dir)
sv = nv*ps  # mm (vertical dir)

DSD = 430  # Distance from the x-ray source to detector (mm)
DSO = 320  # Distance from x-ray source to the axis of rotation (isocenter) (mm)

# Angle settings
direction = 1  # Rotation direction (gantry rotation direction) (1 or -1)
dang = 2  # Angle between captures (degrees)
deg = np.arange(0, 359, dang)  #  List of all capture angles (0-359 by steps of dang)
deg = deg * direction  # Move along angles in the correct rotation direction
num_proj = len(deg)  # Total number of projections

# This parameter is to correct data taken with less than 360 degrees
parker = 0  # Data w/ 360 degree data: parker = 0; otherwise parker = 1

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

# Interpolation type for backprojection
interpolation_type = 'linear'  # Options: 'linear, 'nearest'

# If an Nvidia GPU card is available, this can run reconstructions more quickly (potentially)
# It is necessary to install cupy (pip install cupy)
gpu = 0  # GPU present: gpu = 1, otherwise gpu = 0

