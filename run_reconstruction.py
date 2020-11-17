import os
import numpy as np
from scipy.io import savemat
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
import spectral_reconstruction as spr

path = r'D:\OneDrive - University of Victoria\Research\LDA Data'

folders = glob(os.path.join(path, '*720*'))
air = np.load(os.path.join(path, 'airscan_120kVP_1mA_1mmAl_3x8coll_360s_6frames', 'Data', 'data.npy'))
dark = np.load(os.path.join(path, 'darkscan_360s_6frames', 'Data', 'data.npy'))

nums = [0.25, 0.5, 1]

for f, folder in enumerate(folders):
    savepath = os.path.join(folder, 'CT')
    os.makedirs(savepath, exist_ok=True)
    data720 = np.load(os.path.join(folder, 'Data', 'data.npy'))
    data_shape = np.shape(data720)
    data360 = np.zeros((data_shape[0]//2, *data_shape[1:]))
    data180 = np.zeros((data_shape[0]//4, *data_shape[1:]))

    print(np.shape(data720), np.shape(data360), np.shape(data180))

    for i in np.arange(data_shape[0]//2):
        data360[i] = np.sum(data720[2 * i:2 * i + 2], axis=0)

    for i in np.arange(data_shape[0]//4):
        data180[i] = np.sum(data720[4 * i:4 * i + 4], axis=0)

    data720 = data720[8:8+720]
    data360 = data360[5:5+360]
    data180 = data180[2:2+180]

    print(np.shape(data720), np.shape(data360), np.shape(data180))
    for j in np.arange(1, 6):
        if j > 1:
            temp_dark = np.sum(dark[1:j+1], axis=0)
            temp_air = np.sum(air[1:j+1], axis=0)
        else:
            temp_dark = dark[1]
            temp_air = air[1]

        temp_air = np.subtract(temp_air, temp_dark)
        print(j, nums[f], (60*j/nums[f]), (60*j/(nums[f]*2)), (60*j/(nums[f]*4)))
        proj720 = spr.generate_projections(data720, temp_air/(60*j/nums[f]), temp_dark/(60*j/nums[f]))
        proj360 = spr.generate_projections(data360, temp_air/(60*j/(nums[f]*2)), temp_dark/(60*j/(nums[f]*2)))
        proj180 = spr.generate_projections(data180, temp_air/(60*j/(nums[f]*4)), temp_dark/(60*j/(nums[f]*4)))

        dict720 = {'data': proj720, 'label': 'projections 720'}
        dict360 = {'data': proj360, 'label': 'projections 360'}
        dict180 = {'data': proj180, 'label': 'projections 180'}
        savemat(os.path.join(savepath, f'proj720_{j}-60s.mat'), dict720)
        savemat(os.path.join(savepath, f'proj360_{j}-60s.mat'), dict360)
        savemat(os.path.join(savepath, f'proj180_{j}-60s.mat'), dict180)
# dic = {'data': air, 'label': 'airscan'}
# dic2 = {'data': dark, 'label': 'darkscan'}
# savemat(os.path.join(path, air_folder, 'Data/data.mat'), dic)
# savemat(os.path.join(path, dark_folder, 'Data/data.mat'), dic2)
#
# start = datetime.now().timestamp()
# proj = spr.generate_projections(data, air, dark)
# stop = datetime.now().timestamp()
# print('projections done')
# print(stop-start)
# start = datetime.now().timestamp()
# filt_proj = spr.filtering(proj)
# stop = datetime.now().timestamp()
# print('filtering done')
# print(stop-start)
# ct = spr.CT_backprojection(filt_proj)
# # np.save(path + '/CT_data.npy', ct)
#
# for i in range(24):
#     plt.imshow(ct[6, :, :, i])
#     plt.show()
#     plt.pause(0.5)
#     plt.close()
