import os
import numpy as np
from scipy.io import savemat
from datetime import datetime
import matplotlib.pyplot as plt
import spectral_reconstruction as spr

path = r'D:\OneDrive - University of Victoria\Research\LDA Data'

data_folder = 'ct_test_110920'
air_folder = 'airscan_120kVP_1mA_1mmAl_3x8coll_60s_110920'
dark_folder = 'darkscan_60s'

data = np.load(os.path.join(path, data_folder, 'Data/data.npy'))
air = np.load(os.path.join(path, air_folder, 'Data/data.npy')) / 60
dark = np.load(os.path.join(path, dark_folder, 'Data/data.npy')) / 60

# dic = {'data': air, 'label': 'airscan'}
# dic2 = {'data': dark, 'label': 'darkscan'}
# savemat(os.path.join(path, air_folder, 'Data/data.mat'), dic)
# savemat(os.path.join(path, dark_folder, 'Data/data.mat'), dic2)

start = datetime.now().timestamp()
proj = spr.generate_projections(data, air, dark)
stop = datetime.now().timestamp()
print('projections done')
print(stop-start)
start = datetime.now().timestamp()
filt_proj = spr.filtering(proj)
stop = datetime.now().timestamp()
print('filtering done')
print(stop-start)
ct = spr.CT_backprojection(filt_proj)
# np.save(path + '/CT_data.npy', ct)

for i in range(24):
    plt.imshow(ct[6, :, :, i])
    plt.show()
    plt.pause(0.5)
    plt.close()
