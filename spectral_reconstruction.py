import numpy as np
from scipy.ndimage import map_coordinates
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
    # End up with <capture, asic, row, column, counter> (5D array)
    if num_views == -1:
        data = np.sum(data, axis=1)
        air = np.sum(air, axis=1)
    else:
        data = np.sum(data[:, 0:num_views], axis=1)
        air = np.sum(air[:, 0:num_views], axis=1)

    # Permute to order <counter, capture(angle), asic, row, column>
    data = np.transpose(data, axes=(4, 0, 1, 2, 3))
    air = np.transpose(air, axes=(4, 0, 1, 2, 3))

    # Correct for any non-responsive pixels
    data = correct_dead_pixels(data, param.dead_pixel_mask)
    air = correct_dead_pixels(air, param.dead_pixel_mask)

    # Calculate projections
    proj = -1*np.log(np.divide(data, air))

    # Concatenate the asics to get the full field of view
    data_shape = np.shape(proj)
    num_asics = data_shape[2]  # Find number of asics

    projections = proj[:, :, 0]  # Get the initial asic
    for n in np.arange(1, num_asics):
        next_asic = proj[:, :, n]
        projections = np.concatenate((projections, next_asic), axis=3)  # Concatenate each asic along column axis

    projections = multiple_proj_remove_stripe(projections, 2)  # Remove any striping artifacts from the projection data

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

    # Find the next highest power of 2 of number of pixels horizontally in the detector multiplied by 2
    filt_len = int(np.max([64, 2**np.ceil(np.log2(2*param.nu))]))

    ramp_kernel = ramp_flat(filt_len)  # Calculate the ramp filter kernel

    filt = filter_array(param.filter, ramp_kernel, filt_len)  # Calculate the full filter array
    filt = np.tile(np.reshape(filt, (1, np.size(filt))), (param.nv, 1))  # Copy the filter nv times (nv = number of pixels vertically)

    # For each projection, filter the data
    for i, counter in enumerate(projections):
        for j, proj in enumerate(counter):

            filt_proj = np.zeros([param.nv, filt_len], dtype='float32')
            filt_proj[:, int(filt_len/2-param.nu/2):int(filt_len/2+param.nu/2)] = proj  # Set proj data into the middle nu rows
            filt_proj = fft(filt_proj, axis=1)  # Compute the Fourier transform along each column

            filt_proj = filt_proj * filt  # Apply the filter to the Fourier transform of the data
            filt_proj = np.real(ifft(filt_proj, axis=1))  # Get only the real portion of the inverse Fourier transform

            # Grab the filtered data and apply a correction factor based on the number of projections and system geometry
            proj = filt_proj[:, int(filt_len/2-param.nu/2):int(filt_len/2+param.nu/2)] /2/param.ps * \
                   (2*np.pi/param.num_proj) / 2 * (param.DSD/param.DSO)

            projections[i, j] = proj  # Reassign the unfiltered data as the newly filtered data

    return projections


def ramp_flat(n):
    """
    This function creates the ramp filter array of the correct size based on the projection data
    :param n: int
                The length of the filter based on the data
    :return: 1d array
                The ramp filter of the correct size for the projection data

    Adapted from:  Kyungsang Kim (2020). 3D Cone beam CT (CBCT) projection backprojection FDK, iterative reconstruction
    Matlab examples (https://www.mathworks.com/matlabcentral/fileexchange/35548-3d-cone-beam-ct-cbct-projection-
    backprojection-fdk-iterative-reconstruction-matlab-examples), MATLAB Central File Exchange. Retrieved May 19, 2020.
    """
    nn = np.arange(-n/2, n/2)
    h = np.zeros(np.size(nn), dtype='float32')
    h[int(n/2)] = 0.25  # Set center point (0.0) equal to 1/4
    odd = np.mod(nn, 2) == 1  # odd = False, even = True
    h[odd] = -1 / (np.pi * nn[odd])**2

    return h


def filter_array(filter, kernel, order, d=1):
    """
    This function takes the high pass filter type, ramp filter kernel, the order, and cutoff and calculates the filter
    array to apply to the projection data
    :param filter: String
                High pass filter type, see Parameters.py for options under the 'filter' variable
    :param kernel: 1D numpy array
                The ramp filter kernel
    :param order: int
                The filter length depending on the data
    :param d: float, default = 1
                Cutoff for the high-pass filter. On the range [0, 1]
    :return: 1D numpy array
                The filter array

    Adapted from:  Kyungsang Kim (2020). 3D Cone beam CT (CBCT) projection backprojection FDK, iterative reconstruction
    Matlab examples (https://www.mathworks.com/matlabcentral/fileexchange/35548-3d-cone-beam-ct-cbct-projection-
    backprojection-fdk-iterative-reconstruction-matlab-examples), MATLAB Central File Exchange. Retrieved May 19, 2020.
    """
    f_kernel = np.abs(fft(kernel))*2
    filt = f_kernel[0:int(order/2+1)]
    w = 2*np.pi*np.arange(len(filt))/order  # Frequency axis up to Nyquist

    if filter is 'shepp-logan':
        # Be aware of your d value - do not set d equal to zero
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

    for iz in np.arange(param.nz):

        pv = ((param.zs[iz] * param.DSD / (ry + param.DSO)) - param.vs[0]) / param.ps + 1
        coords = np.array([np.ravel(pv), np.ravel(pu)])
        vol[:, :, iz] = ratio * np.reshape(map_coordinates(projection, coords, mode='nearest'), (param.ny, param.nx))

    vol[np.isnan(vol)] = 0
    return vol


def correct_dead_pixels(data, dead_pixel_mask):
    """
    This is to correct for known dead pixels. Takes the average of the eight surrounding pixels.
    Could implement a more sophisticated algorithm here if needed.
    :param data: 5D numpy array
                The data array in which to correct the pixels <counter, captures, asics, rows, columns>
    :param dead_pixel_mask: 3D numpy array
                Mask with value 1 at good pixel coordinates and nan at bad pixel coordinates <asic, row, column>
    :return: The data array corrected for the dead pixels
    """
    dead_pixels = np.array(np.argwhere(np.isnan(dead_pixel_mask)), dtype='int')  # Find the dead pixels (i.e = to nan)
    data_shape = np.shape(data)
    for coords in dead_pixels:
        asic = coords[0]  # The asic number
        pixel = coords[1:]  # The row, column coordinates
        for i in np.arange(data_shape[0]):
            for j in np.arange(data_shape[1]):
                # Pixel is corrected in every counter and capture
                avg_val = get_average_pixel_value(data[i, j, asic], pixel, dead_pixel_mask[asic])
                data[i, j, asic, pixel[0], pixel[1]] = avg_val  # Set the new value in the 5D array

    return data


def get_average_pixel_value(img, pixel, dead_pixel_mask):
    """
    Averages the dead pixel using the 8 nearest neighbours
    Checks the dead pixel mask to make sure each of the neighbors is not another dead pixel
    :param img: 2D array
                The projection image
    :param pixel: tuple (row, column)
                The problem pixel (is a 2-tuple)
    :param dead_pixel_mask: 2D numpy array
                Mask with 1 at good pixel coordinates and nan at bad pixel coordinates
    :return:
    """
    shape = np.shape(img)
    row, col = pixel

    # Grabs each of the neighboring pixel values and sets to nan if they are bad pixels or
    # outside the bounds of the image
    if col == shape[1]-1:
        n1 = np.nan
    else:
        n1 = img[row, col+1] * dead_pixel_mask[row, col+1]
    if col == 0:
        n2 = np.nan
    else:
        n2 = img[row, col-1] * dead_pixel_mask[row, col-1]
    if row == shape[0]-1:
        n3 = np.nan
    else:
        n3 = img[row+1, col] * dead_pixel_mask[row+1, col]
    if row == 0:
        n4 = np.nan
    else:
        n4 = img[row-1, col] * dead_pixel_mask[row-1, col]
    if col == shape[1]-1 or row == shape[0]-1:
        n5 = np.nan
    else:
        n5 = img[row+1, col+1] * dead_pixel_mask[row+1, col+1]
    if col == 0 or row == shape[0]-1:
        n6 = np.nan
    else:
        n6 = img[row+1, col-1] * dead_pixel_mask[row+1, col-1]
    if col == shape[1]-1 or row == 0:
        n7 = np.nan
    else:
        n7 = img[row-1, col+1] * dead_pixel_mask[row-1, col+1]
    if col == 0 or row == 0:
        n8 = np.nan
    else:
        n8 = img[row-1, col-1] * dead_pixel_mask[row-1, col-1]

    # Takes the average of the neighboring pixels excluding nan values
    avg = np.nanmean(np.array([n1, n2, n3, n4, n5, n6, n7, n8]))

    return avg


def multiple_proj_remove_stripe(images, level, wname='db5', sigma=1.2):
    """
    Calls the remove stripe function multiple times for the number of 2d projections images in the 4d data
    :param images: 4D numpy array
                The projection image data. Shape <counters, captures, rows, columns>
    :param level: int
                The highest decomposition level.
    :param wname: str, optional
                The wavelet type. Default value is 'db5'
    :param sigma: float, optional
                The damping factor in Fourier space. Default value is '1.2'
    :return: 4D numpy array
                The resulting filtered images.
    """
    # Rearrange so that in the second for loop each image (sinogram) is angles vs. columns
    images = np.transpose(images, axes=(0, 2, 1, 3))
    for i, energybin in enumerate(images):
        for j, img in enumerate(energybin):
            img = remove_stripe(img, level, wname=wname, sigma=sigma)  # Remove the stripes in one direction
            images[i, j, :, :] = img
    images = np.transpose(images, axes=(0, 2, 1, 3))  # Rearrange back to the proper orientation

    return images


def remove_stripe(img, level, wname='db5', sigma=1.2):
    """
    Suppress vertical stripe artifacts using the Fourier-Wavelet based method by Munch et al.
    :param img: 2d array
                The two-dimensional array representing the image or the sinogram to de-stripe.
    :param level: int
                The highest decomposition level.
    :param wname: str, optional
                The wavelet filter. Default value is 'db5'. Check pywavelets for different possible filters
                db is Daubechies, and the level goes from 2-38
    :param sigma: float, optional
                The damping factor in Fourier space. Default value is '1.2'
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


