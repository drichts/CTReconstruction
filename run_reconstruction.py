import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import spectral_reconstruction as spr

data = np.load(r'C:\Users\Neusoft\Desktop\LDA\Devon\ct_test_110920\Data\data.npy')
air = np.load(r'C:\Users\Neusoft\Desktop\LDA\airscan_120kVP_1mA_1mmAl_3x8coll_60s\Data\data.npy')
dark = np.load(r'C:\Users\Neusoft\Desktop\LDA\darkscan_60s\Data\data.npy')

dic = {'data': data, 'label': 'projections'}
dicair = {'data': air, 'label': 'air'}
dicdark = {'data': dark, 'label': 'dark'}
savemat(r'C:\Users\Neusoft\Desktop\LDA\Devon\ct_test_110920\Data\data.mat', dic)
savemat(r'C:\Users\Neusoft\Desktop\LDA\airscan_120kVP_1mA_1mmAl_3x8coll_60s\Data\data.mat', dicair)
savemat(r'C:\Users\Neusoft\Desktop\LDA\darkscan_60s\Data\data.mat', dicdark)


# proj = spr.generate_projections(data, air, dark)
# print('projections done')
# filt_proj = spr.filtering(proj)
# print('filtering done')
# ct = spr.CT_backprojection(filt_proj)
# np.save(r'C:\Users\Neusoft\Desktop\LDA\Data\test1\CT.npy', ct)
# ct = np.load(r'C:\Users\Neusoft\Desktop\LDA\Data\test1\CT.npy')
#
# plt.imshow(ct[6, :, :, 15])
# plt.show()
