import numpy as np
import matplotlib.pyplot as plt
import spectral_reconstruction as spr

data = np.load(r'C:\Users\Neusoft\Desktop\LDA Data\phantom.npy')
air = np.load(r'C:\Users\Neusoft\Desktop\LDA Data\airscan_0-5s.npy')
dark = np.load(r'C:\Users\Neusoft\Desktop\LDA Data\darkscan_0-5s.npy')

proj = spr.generate_projections(data, air, dark)
print('projections done')
filt_proj = spr.filtering(proj)
print('filtering done')
ct = spr.CT_backprojection(filt_proj)
np.save(r'C:\Users\Neusoft\Desktop\LDA Data\CT.npy', ct)

for i in range(24):
    plt.imshow(ct[6, :, :, i])
    plt.show()
    plt.pause(0.5)
    plt.close()
