import numpy as np
from scipy.io import loadmat
import glob
import re

path = 'D:/Research/Python Data/Redlen/Test Projections/'
files = glob.glob(path + '*filtered.mat')
files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

proj = np.zeros([13, 120, 24, 180])

for i, file in enumerate(files):
    data = loadmat(file)['proj_filtered']
    proj[i, :, :, :] = data

proj = np.transpose(proj, axes=(0, 3, 2, 1))

#np.save(path + 'test_projections.npy', proj)
np.save(path + 'test_projections_filtered.npy', proj)


