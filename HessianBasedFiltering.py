import numpy as np
from scipy import ndimage


def get_hessian(image, sigma):
    
    # create kernel based on 3x3 kernel and the scale
    x, y = np.mgrid[-np.round(sigma*3):np.round(sigma*3), -np.round(sigma*3):np.round(sigma*3)]
    
    kernel_dxx = 1/(2*np.pi*sigma**4) * (np.square(x)/sigma**2 - 1) * np.exp(-(np.square(x) + np.square(y))/(2*sigma**2))
    kernel_dxy = 1/(2*np.pi*sigma**6) * x * y * np.exp(-(np.square(x) + np.square(y))/(2*sigma**2))
    kernel_dyy = kernel_dxx.T

    Dxx = ndimage.convolve(image, kernel_dxx, mode='constant', cval=1.0)
    Dxy = ndimage.convolve(image, kernel_dxy, mode='constant', cval=1.0)
    Dyy = ndimage.convolve(image, kernel_dyy, mode='constant', cval=1.0)
    
    return Dxx, Dxy, Dyy


def get_eigen_values(Dxx, Dxy, Dyy):

    tmp = np.sqrt(np.square(Dxx - Dyy) + 4*np.square(Dxy))

    mu1 = 0.5*(Dxx + Dyy + tmp)
    mu2 = 0.5*(Dxx + Dyy - tmp)

    check = np.abs(mu1) > np.abs(mu2)

    lambda1 = mu1
    lambda1[check] = mu2[check]
    lambda2 = mu2
    lambda2[check] = mu1[check]

    return lambda1, lambda2


def eigen_value_filter(image, s1, s2):

    # make sure image is of the correct type
    image = image.astype('float32')
    sigmas = [s1, s2]
    multi_scale_image = np.zeros((2, image.shape[0], image.shape[1]))
    
    # iterate through the different scale and calculate the filter value
    for i, sigma in enumerate(sigmas):
        
        # compute hessian by filtering with 2nd derivative of Gaussian
        Dxx, Dxy, Dyy = get_hessian(image, sigma)
    
        # correct for scale
        Dxx = (sigma**2)*Dxx
        Dxy = (sigma**2)*Dxy
        Dyy = (sigma**2)*Dyy
        
        # compute eigenvalues of hessian
        lambdas = get_eigen_values(Dxx, Dxy, Dyy)
        filt = np.abs(np.min([lambdas], axis=1))
        
        multi_scale_image[i] = filt

    # calculate maximum across multiple scales
    filtered_image = np.max(multi_scale_image, axis=0)
    return filtered_image
