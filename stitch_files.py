import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
from scipy.io import loadmat
import spectral_reconstruction as sct

def get_data():
    dead_pixel_mask = np.load('D:/Research/Python Data/dead_pixel_mask.npy')

    path = r'D:Research\sCT Scan Data\AuGd_width_14_12-2-19/'
    airpath = r'D:\Research\sCT Scan Data\AuGd_width_14_12-2-19\Air/'
    darkpath = r'D:\Research\sCT Scan Data\dark_update\Raw Test Data\M15691\UNIFORMITY/'
    path2 = '/Raw Test Data/M15691/STABILITY/'

    run = 16

    nor = 24
    noc = 36
    noa = 180

    air0 = np.zeros([13, 12, 2, nor, noc])
    air1 = np.zeros([13, 12, 2, nor, noc])
    air2 = np.zeros([13, 12, 2, nor, noc])

    air_subdir = glob.glob(airpath + '*')

    airfiles0a = glob.glob(air_subdir[0] + path2 + '*A0*')
    airfiles0b = glob.glob(air_subdir[0] + path2 + '*A1*')
    airfiles1a = glob.glob(air_subdir[1] + path2 + '*A0*')
    airfiles1b = glob.glob(air_subdir[1] + path2 + '*A1*')
    airfiles2a = glob.glob(air_subdir[2] + path2 + '*A0*')
    airfiles2b = glob.glob(air_subdir[2] + path2 + '*A1*')

    for i in np.arange(12):
        temp0 = np.squeeze(loadmat(airfiles0a[i+10])['cc_struct']['data'][0][0][0][0][0])
        temp1 = np.squeeze(loadmat(airfiles0b[i+10])['cc_struct']['data'][0][0][0][0][0])
        temp2 = np.squeeze(loadmat(airfiles1a[i+10])['cc_struct']['data'][0][0][0][0][0])
        temp3 = np.squeeze(loadmat(airfiles1b[i+10])['cc_struct']['data'][0][0][0][0][0])
        temp4 = np.squeeze(loadmat(airfiles2a[i+10])['cc_struct']['data'][0][0][0][0][0])
        temp5 = np.squeeze(loadmat(airfiles2b[i+10])['cc_struct']['data'][0][0][0][0][0])

        air0[:, i, 0] = np.add(air0[:, i, 0], temp0)
        air0[:, i, 1] = np.add(air0[:, i, 1], temp1)
        air1[:, i, 0] = np.add(air1[:, i, 0], temp2)
        air1[:, i, 1] = np.add(air1[:, i, 1], temp3)
        air2[:, i, 0] = np.add(air2[:, i, 0], temp4)
        air2[:, i, 1] = np.add(air2[:, i, 1], temp5)

    dark = np.zeros([13, 1, 2, 24, 36])
    dark[:, 0, 0] = np.squeeze(loadmat(glob.glob(darkpath + '*A0_Run016*')[0])['cc_struct']['data'][0][0][0][0][0])
    dark[:, 0, 1] = np.squeeze(loadmat(glob.glob(darkpath + '*A1_Run016*')[0])['cc_struct']['data'][0][0][0][0][0])

    air0 = sct.correct_dead_pixels(air0, dead_pixel_mask)
    air1 = sct.correct_dead_pixels(air1, dead_pixel_mask)
    air2 = sct.correct_dead_pixels(air2, dead_pixel_mask)

    dark = sct.correct_dead_pixels(dark, dead_pixel_mask)

    air0 = np.subtract(air0, dark)
    air1 = np.subtract(air1, dark)
    air2 = np.subtract(air2, dark)

    air0 = np.mean(air0, axis=1, keepdims=True)
    air1 = np.mean(air1, axis=1, keepdims=True)
    air2 = np.mean(air2, axis=1, keepdims=True)

    datapath = glob.glob(path + '*Rot*')

    files0 = glob.glob(datapath[0] + path2 + '*A0*')
    files1 = glob.glob(datapath[0] + path2 + '*A1*')
    files2 = glob.glob(datapath[1] + path2 + '*A0*')
    files3 = glob.glob(datapath[1] + path2 + '*A1*')
    files4 = glob.glob(datapath[2] + path2 + '*A0*')
    files5 = glob.glob(datapath[2] + path2 + '*A1*')

    proj0 = np.zeros([13, 180, 2, 24, 36])
    proj1 = np.zeros([13, 180, 2, 24, 36])
    proj2 = np.zeros([13, 180, 2, 24, 36])

    for a in np.arange(noa):
        if a > 190:
            proj0[:, a, 0] = np.squeeze(loadmat(files0[a + run+1])['cc_struct']['data'][0][0][0][0][0])
            proj0[:, a, 1] = np.squeeze(loadmat(files1[a + run+1])['cc_struct']['data'][0][0][0][0][0])
        else:
            proj0[:, a, 0] = np.squeeze(loadmat(files0[a + run])['cc_struct']['data'][0][0][0][0][0])
            proj0[:, a, 1] = np.squeeze(loadmat(files1[a + run])['cc_struct']['data'][0][0][0][0][0])
        proj1[:, a, 0] = np.squeeze(loadmat(files2[a + run])['cc_struct']['data'][0][0][0][0][0])
        proj1[:, a, 1] = np.squeeze(loadmat(files3[a + run])['cc_struct']['data'][0][0][0][0][0])
        proj2[:, a, 0] = np.squeeze(loadmat(files4[a + run])['cc_struct']['data'][0][0][0][0][0])
        proj2[:, a, 1] = np.squeeze(loadmat(files5[a + run])['cc_struct']['data'][0][0][0][0][0])

    proj0 = sct.correct_dead_pixels(proj0, dead_pixel_mask)
    proj1 = sct.correct_dead_pixels(proj1, dead_pixel_mask)
    proj2 = sct.correct_dead_pixels(proj2, dead_pixel_mask)

    proj0 = np.subtract(proj0, dark)
    proj1 = np.subtract(proj1, dark)
    proj2 = np.subtract(proj2, dark)

    proj0 = -1*np.log(np.divide(proj0, air0))
    proj1 = -1*np.log(np.divide(proj1, air1))
    proj2 = -1*np.log(np.divide(proj2, air2))

    proj0 = np.concatenate((np.flip(np.flip(proj0[:, :, 1], axis=3), axis=2), proj0[:, :, 0]), axis=3)
    proj1 = np.concatenate((np.flip(np.flip(proj1[:, :, 1], axis=3), axis=2), proj1[:, :, 0]), axis=3)
    proj2 = np.concatenate((np.flip(np.flip(proj2[:, :, 1], axis=3), axis=2), proj2[:, :, 0]), axis=3)

    return proj0, proj1, proj2

proj0, proj1, proj2 = get_data()


pp = np.concatenate((proj0, proj1), axis=3)
pp = np.concatenate((pp, proj2), axis=3)
pp = np.delete(pp, np.s_[49:81], axis=3)
pp = np.delete(pp, np.s_[85:117], axis=3)
np.save('D:/Research/Python Data/Redlen/proj_py_default.npy', pp)
