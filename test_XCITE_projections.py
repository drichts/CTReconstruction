import numpy as np
from scipy.io import loadmat
import glob
import re

path = 'D:/Research/Python Data/Redlen/Test Projections/'
files = glob.glob(path + '*proj.mat')

files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

proj = np.zeros([13, 119, 24, 180])

for i, file in enumerate(files):
    data = loadmat(file)['proj_img']
    proj[i, :, :, :] = data
proj = np.transpose(proj, axes=(0, 3, 2, 1))
np.save(path + 'test_proj.npy', proj)


proj_stripe = np.zeros([13, 119, 24, 180])
files = glob.glob(path + '*stripcor.mat')

files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

for i, file in enumerate(files):
    data = loadmat(file)['proj_cor']
    proj_stripe[i, :, :, :] = data
proj_stripe = np.transpose(proj_stripe, axes=(0, 3, 2, 1))
np.save(path + 'test_proj_stripe.npy', proj_stripe)


proj_filt = np.zeros([13, 119, 24, 180])
files = glob.glob(path + '*filt.mat')

files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

for i, file in enumerate(files):
    data = loadmat(file)['proj_filtered']
    proj_filt[i, :, :, :] = data
proj_filt = np.transpose(proj_filt, axes=(0, 3, 2, 1))
np.save(path + 'test_proj_filt.npy', proj_filt)



