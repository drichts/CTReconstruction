import numpy as np
#import cupy as cp
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import glob
import pywt
import Parameters as param
from numpy.fft import fftshift, ifftshift, fft, ifft


def generate_projections(data, air, dark, num_views=-1):
    """
    This function takes the captured data and calculates a projection at each capture (angle) projection = -ln(I/I0)
    Where I is the data intensity taken at that angle, and I0 is the airscan intensity

    :param data: 6D numpy array <captures, views, asics, rows, columns, counters>
                The data array. Assume the asics are in the correct orientation relative to the others

    :param air: 6D numpy array <1 capture, views, asics, rows, columns, counters>
                The airscan data array. The number of views and view duration should be the same as for 1 capture of
                the 'data' array

    :param dark: 6D numpy array <1 capture, views, asics, rows, columns, counters>
                The darkfield data array. The # of views and view duration should be the same as for 1 capture of the
                'data' array

    :param num_views: int
                The number of views to sum to obtain the desired capture time length
                Default: -1 (sum all views)

    :return: 4D numpy array <counters, captures, rows, columns>
                The calculated projection data with dead pixels corrected and stripes removed
    """

    np.seterr(divide='ignore', invalid='ignore')

    # Correct the data and air scans for the dark current (may not strictly be necessary)
    data = np.subtract(data, dark)
    air = np.subtract(air, dark)

    # Sum the desired number of views for each capture (default is to sum all views)
    if num_views == -1:
        data = np.sum(data, axis=1)
        air = np.sum(air, axis=1)
    else:
        data = np.sum(data[:, 0:num_views], axis=1)
        air = np.sum(air[:, 0:num_views], axis=1)

    # Calculate projections
    proj = -1*np.log(np.divide(data, air))

    # Correct for any non-responsive pixels
    #proj = correct_dead_pixels(proj)

    # Concatenate the asics to get the full field of view
    data_shape = np.shape(proj)
    num_asics = data_shape[2]  # Find number of asics

    projections = proj[:, :, 0, :, :, :]  # Get the initial asic
    for n in np.arange(1, num_asics):
        next_asic = proj[:, :, n, :, :, :]
        projections = np.concatenate(projections, next_asic, axis=4)  # Concatenate each asic along column axis

    # Permute projections so x, y-direction order lines up with column, row order for ease of filtering and backprojection
    projections = np.transpose(projections, axes=(3, 0, 1, 2))

    # Remove any striping artifacts from the projection data
    # projections = multiple_image_remove_stripe(projections, 2)

    return projections


def filtering(projections):
    """
    Applies a ramp filter at low frequencies and the desired high-pass filter

    :param projections: 4D numpy array
                The projection data. Shape: <counters, captures, rows, columns>

    :return: 4D numpy array
                The filtered projection data. Shape: <counters, captures, rows, columns>

    Adapted from:  Kyungsang Kim (2020). 3D Cone beam CT (CBCT) projection backprojection FDK, iterative reconstruction
    Matlab examples (https://www.mathworks.com/matlabcentral/fileexchange/35548-3d-cone-beam-ct-cbct-projection-
    backprojection-fdk-iterative-reconstruction-matlab-examples), MATLAB Central File Exchange. Retrieved May 19, 2020.
    """

    uu, vv = np.meshgrid(param.us, param.vs)  # Create a meshgrid of x, y coordinates of all pixels

    w = param.DSD / np.sqrt(param.DSD**2 + uu**2 + vv**2)  # Correction for each point based on distance from source to the coordinate

    projections = np.multiply(projections, w)  # Correct each projection angle for detector flatness

    if param.parker == 1:
        pass  # Correct for Parker Weighting

    filt_len = int(np.max([64, 2**np.ceil(np.log2(2*param.nu))]))

    ramp_kernel = ramp_flat(filt_len)  # Calculate the ramp filter kernel

    d = 1  # Cut off (0~1)
    filt = filter_array(param.filter, ramp_kernel, filt_len, d)  # Calculate the full filter array
    filt = np.tile(np.reshape(filt, (1, np.size(filt))), (param.nv, 1))  # Copy the filter nv times (nv = number of pixels vertically)

    # For each projection, filter the data
    for idx, proj in enumerate(projections):

        filt_proj = np.zeros([param.nv, filt_len])
        filt_proj[:, int(filt_len/2-param.nu/2):int(filt_len/2+param.nu/2)] = proj  # Set proj data into the middle nu rows
        filt_proj = fft(filt_proj, axis=1)  # Compute the Fourier transform along each column

        filt_proj = filt_proj * filt  # Apply the filter to the Fourier transform of the data
        filt_proj = np.real(ifft(filt_proj, axis=1))  # Get only the real portion of the inverse Fourier transform

        # Grab the filtered data and apply a correction factor based on the number of projections and system geometry
        if param.parker == 1:
            proj = filt_proj[:, int(filt_len/2-param.nu/2):int(filt_len/2+param.nu/2)] / \
                   2/param.ps * (2*np.pi / (180/param.dang)) / 2 * (param.DSD/param.DSO)
        else:
            proj = filt_proj[:, int(filt_len/2-param.nu/2):int(filt_len/2+param.nu/2)] /\
                   2/param.ps * (2*np.pi/param.num_proj) / 2 * (param.DSD/param.DSO)

        projections[idx] = proj  # Reassign the unfiltered data as the newly filtered data

    return projections


def ramp_flat(n):
    """

    :param n:
    :return:
    Adapted from:  Kyungsang Kim (2020). 3D Cone beam CT (CBCT) projection backprojection FDK, iterative reconstruction
    Matlab examples (https://www.mathworks.com/matlabcentral/fileexchange/35548-3d-cone-beam-ct-cbct-projection-
    backprojection-fdk-iterative-reconstruction-matlab-examples), MATLAB Central File Exchange. Retrieved May 19, 2020.
    """
    nn = np.arange(-n/2, n/2)
    h = np.zeros(np.size(nn))
    h[int(n/2)] = 0.25  # Set center point (0.0) equal to 1/4
    odd = np.mod(nn, 2) == 1  # odd = False, even = True
    h[odd] = -1 / (np.pi * nn[odd])**2

    return h


def filter_array(filter, kernel, order, d):
    """
    This function takes the high pass filter type, ramp filter kernel, the order, and cutoff and calculates the filter
    array to apply to the projection data
    :param filter: high pass filter type, see Parameters.py for options under the 'filter' variable
    :param kernel: the ramp filter kernel
    :param order:
    :param d:
    :return filt: The filter array
    Adapted from:  Kyungsang Kim (2020). 3D Cone beam CT (CBCT) projection backprojection FDK, iterative reconstruction
    Matlab examples (https://www.mathworks.com/matlabcentral/fileexchange/35548-3d-cone-beam-ct-cbct-projection-
    backprojection-fdk-iterative-reconstruction-matlab-examples), MATLAB Central File Exchange. Retrieved May 19, 2020.
    """
    f_kernel = np.abs(fft(kernel))*2
    filt = f_kernel[0:int(order/2+1)]
    w = 2*np.pi*np.arange(len(filt))/order

    if filter is 'shepp-logan':
        # Be aware of your d value - do not set to zero
        filt[2:] = filt[2:] * np.sin(w[2:]/(2*d)) / (w[2:]/(2*d))
    elif filter is 'cosine':
        filt[2:] = filt[2:] * np.cos(w[2:]/(2*d))
    elif filter is 'hamming':
        filt[2:] = filt[2:] * (0.54 + 0.46 * np.cos(w[2:]/d))
    elif filter is 'hann':
        filt[2:] = filt[2:] * (1 + np.cos(w[2:]/d)) / 2
    else:
        print(filter)
        raise Exception('Filter type not recognized.')

    filt[w > np.pi*d] = 0  # Crop the frequency response
    filt = np.concatenate((filt, np.flip(filt[1:-1])))  # Make the filter symmetric

    return filt


def CT_backprojection(projections):
    """
    This function takes the projection data and outputs the CT image

    :param projections: 4D numpy array
                The projection data. Shape <counter, capture, row, column>

    :return: 4D numpy array
                The CT image. Shape <counter, x, y, z>, where z is along the axial direction
    """
    num_counters = len(projections)  # Get the number of counters

    # The empty array for the CT volume
    image = np.zeros([num_counters, param.nx, param.ny, param.nz])

    # Go through each energy bin in succession
    for counter, energydata in enumerate(projections):
        # Go through every projection angle
        for angle, proj in enumerate(energydata):
            new_data = backprojection(proj, angle)  # Get the backprojection of the current angle
            image[counter] = np.add(image[counter], new_data)  # Add it to the volume

    return image


def backprojection(projection, proj_num):
    """
    Takes a single projection angle and backprojects the data into an image of the desired volume

    :param projection: 2D numpy array
                The projection data for the current angle

    :param proj_num: int
                The current angle (in degrees)

    :return: 3D numpy array
                The backprojected data

    Adapted from:  Kyungsang Kim (2020). 3D Cone beam CT (CBCT) projection backprojection FDK, iterative reconstruction
    Matlab examples (https://www.mathworks.com/matlabcentral/fileexchange/35548-3d-cone-beam-ct-cbct-projection-
    backprojection-fdk-iterative-reconstruction-matlab-examples), MATLAB Central File Exchange. Retrieved May 19, 2020.
    """
    angle_rad = param.deg[proj_num] / 360 * 2 * np.pi  # The current projection angle (in radians)
    vol = np.zeros([param.nx, param.ny, param.nz])  # Empty array to hold the backprojected image

    [xx, yy] = np.meshgrid(param.xs, param.ys)  # Create a meshgrid of the center coordinates of the voxels

    # Transpose the xx, yy meshgrid into coordinates measured from axes based on the current angle
    rx = xx * np.cos(angle_rad - np.pi / 2) + yy * np.sin(angle_rad - np.pi / 2)
    ry = -xx * np.sin(angle_rad - np.pi / 2) + yy * np.cos(angle_rad - np.pi / 2)

    pu = ((rx * param.DSD / (ry + param.DSO)) + param.us[0]) / (-param.ps) + 1
    ratio = param.DSO**2 / (param.DSO + ry)**2

    # Uncomment import cupy above to use, and comment out pass
    if param.gpu == 1:
        pass
        #pu = cp.array(pu)
        #cp.cuda.Stream.null.synchronize()
        #proj = cp.array(proj)
        #cp.cuda.Stream.null.synchronize()
        #Ratio = cp.array(Ratio)
        #cp.cuda.Stream.null.synchronize()

    for iz in np.arange(param.nz):

        # Uncomment import cupy above to use, and comment out pass
        if param.gpu == 1:
            pass
            #pv = cp.array(((param.zs[iz] * param.DSD / (ry + param.DSO)) - param.vs[0]) / param.ps + 1)
            #cp.cuda.Stream.null.synchronize()
            #vol[:,:, iz] = (ratio * interp2d(pu, pv, projection, kind=param.interpolation_type))
        else:
            pv = ((param.zs[iz] * param.DSD / (ry + param.DSO)) - param.vs[0]) / param.dv + 1
            vol[:, :, iz] = (ratio * interp2d(pu, pv, projection, kind=param.interpolation_type))

    vol[np.isnan(vol)] = 0

    return vol


def correct_dead_pixels(data, dead_pixels=[]):
    """
    This is to correct for known dead pixels. Takes the average of the eight surrounding pixels.
    Could implement a more sophisticated algorithm here if needed.
    :param data: The full data array
    :param dead_pixels: Array of known dead pixel indices, shape: <(asic, row, column)>
    :return: The data array corrected for the dead pixels
    """
    for pixel in dead_pixels:
        get_average_pixel_value(data, pixel)

    return data


def get_average_pixel_value(img, pixel):
    """
    Averages the dead pixel using the 8 nearest neighbours
    :param img: 2D array
                The projection image
    :param pixel: tuple (row, column)
                The problem pixel (is a 2-tuple)
    :return:
    """
    shape = np.shape(img)
    row, col = pixel

    if col == shape[1]-1:
        n1 = np.nan
    else:
        n1 = img[row, col+1]
    if col == 0:
        n2 = np.nan
    else:
        n2 = img[row, col-1]
    if row == shape[0]-1:
        n3 = np.nan
    else:
        n3 = img[row+1, col]
    if row == 0:
        n4 = np.nan
    else:
        n4 = img[row-1, col]
    if col == shape[1]-1 or row == shape[0]-1:
        n5 = np.nan
    else:
        n5 = img[row+1, col+1]
    if col == 0 or row == shape[0]-1:
        n6 = np.nan
    else:
        n6 = img[row+1, col-1]
    if col == shape[1]-1 or row == 0:
        n7 = np.nan
    else:
        n7 = img[row-1, col+1]
    if col == 0 or row == 0:
        n8 = np.nan
    else:
        n8 = img[row-1, col-1]

    avg = np.nanmean(np.array([n1, n2, n3, n4, n5, n6, n7, n8]))

    return avg


def multiple_proj_remove_stripe(images, level, wname='db5', sigma=1.5):
    """
    Calls the remove stripe function multiple times for the number of 2d projections images in the 4d data
    :param images: 4D numpy array
                The projection image data. Shape <counters, captures, rows, columns>
    :param level: int
                The highest decomposition level.
    :param wname: str, optional
                The wavelet type. Default value is 'db5'
    :param sigma: float, optional
                The damping factor in Fourier space. Default value is '1.5'
    :return: 4D numpy array
                The resulting filtered images.
    """

    for i, energybin in enumerate(images):
        for j, img in enumerate(energybin):
            img = np.rot90(img, axes=(0, 1))  # Rotate the image so the horizontal stripes are now vertical
            img = remove_stripe(img, level, wname=wname, sigma=sigma)  # Remove the stripes
            images[i, j] = np.rot90(img, axes=(1, 0))  # Rotate the image back and replace the uncorrected image

    return images


def remove_stripe(img, level, wname='db5', sigma=1.5):
    """
    Suppress vertical stripe artifacts using the Fourier-Wavelet based method by Munch et al.
    :param img: 2d array
                The two-dimensional array representing the image or the sinogram to de-stripe.
    :param level: int
                The highest decomposition level.
    :param wname: str, optional
                The wavelet type. Default value is 'db5'
    :param sigma: float, optional
                The damping factor in Fourier space. Default value is '1.5'
    :return: 2d array
                The resulting filtered image.
    References
    ----------
    .. [2] B. Munch, P. Trtik, F. Marone, M. Stampanoni, Stripe and ring artifact removal with
           combined wavelet-Fourier filtering, Optics Express 17(10):8567-8591, 2009.
    """

    nrow, ncol = np.shape(img)

    cH = []  # Horizontal detail coefficients
    cV = []  # Vertical detail coefficients
    cD = []  # Diagonal detail coefficients

    # Wavelet decomposition.
    for i in np.arange(level):
        img, (cHi, cVi, cDi) = pywt.dwt2(img, wname)
        cH.append(cHi)
        cV.append(cVi)
        cD.append(cDi)

    # FFT transform of horizontal frequency bands
    for i in np.arange(level):
        # FFT
        fcV = fftshift(fft(cV[i], axis=0))
        my, mx = np.shape(fcV)

        # Damping of vertical stripe information
        yy2 = (np.arange(-np.floor(my/2), -np.floor(my/2) + my)) ** 2
        damp = 1 - np.exp(- yy2 / (2.0 * (sigma**2)))
        fcV = fcV * np.tile(np.reshape(damp, (np.size(damp), 1)), (1, mx))

        # inverse FFT
        cV[i] = np.real(ifft(ifftshift(fcV), axis=0))

    # Wavelet reconstruction
    for i in np.arange(level-1, -1, -1):
        img = img[0:np.shape(cH[i])[0], 0:np.shape(cH[i])[1]]
        img = pywt.idwt2((img, (cH[i], cV[i], cD[i])), wname)

    return img[0:nrow, 0:ncol]
