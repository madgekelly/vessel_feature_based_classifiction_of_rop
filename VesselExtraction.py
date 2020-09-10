import HessianBasedFiltering as hbf
import numpy as np
from skimage import morphology
import cv2
from cv2.ximgproc import guidedFilter
from medpy.filter.smoothing import anisotropic_diffusion


########################### IMAGE ENHANCEMENT ###########################

def guided_filter_enhancement(image, eps_ie, b_ie, w, r_ie, sig_ie):
    # guided filter
    gf1 = guidedFilter(image, image, radius=r_ie, eps=eps_ie)
    # gaussian filter with 'large' kernel size
    gf2 = cv2.GaussianBlur(gf1, (b_ie, b_ie), sig_ie)
    gf3 = cv2.subtract(gf1, gf2)
    # get guided filter
    gf4 = cv2.addWeighted(gf3, w, gf1, 1, 0)
    return gf4


def pre_process_image(image, eps_ie, b_ie, w, n, kap, gam, r_ie, sig_ie):
    inverted_green_channel = 255 - image[:, :, 1]
    gf = guided_filter_enhancement(inverted_green_channel, eps_ie, b_ie, w, r_ie, sig_ie)
    preprocessed_image = anisotropic_diffusion(gf, niter=n, kappa=kap, gamma=gam, option=3)
    return preprocessed_image


########################### THICK VESSEL EXTRACTION ###########################

# returns binarised image of thick vessels
def extract_thick_vessels(image, s_tk, scale_tk1, scale_tk2, o_tk):
    # thick vessel background subtraction
    structure_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s_tk, s_tk))
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, structure_element)
    white_top_hat_image = cv2.subtract(image, opened_image)
    # hessian filtering
    hf_image = hbf.eigen_value_filter(white_top_hat_image, scale_tk1, scale_tk2)
    # Otsu's thresholding
    opt_threshold, thick_vessels = cv2.threshold(cv2.convertScaleAbs(hf_image), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # pre process using area thresholding
    final_image = morphology.remove_small_objects(thick_vessels.astype(bool), min_size=o_tk)
    return final_image.astype('uint8')

###############################################################################


########################### THIN VESSEL EXTRACTION ############################

def detect_end_points(image):
    diagonal = np.array([[-1, -1, -1],[-1, 1, -1],[1, -1, -1]])
    straight = np.array([[-1, -1, -1],[-1, 1, -1],[-1, 1, -1]])
    edges = np.zeros(image.shape).astype('uint8')
    for i in range(4):
        structuring_element = np.rot90(diagonal, i)
        hit_or_miss = cv2.morphologyEx(image, cv2.MORPH_HITMISS, structuring_element)
        edges = cv2.bitwise_or(edges, hit_or_miss)
    for i in range(4):
        structuring_element = np.rot90(straight, i)
        hit_or_miss = cv2.morphologyEx(image, cv2.MORPH_HITMISS, structuring_element)
        edges = cv2.bitwise_or(edges, hit_or_miss)
    return edges


def modified_closing(image, se_size):
    structure_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    structure_element2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*se_size + 1, 2*se_size + 1))
    # skeletonise image
    skeleton = morphology.skeletonize(image).astype('uint8')
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, structure_element)
    end_points = detect_end_points(skeleton)
    region_to_close = cv2.filter2D(end_points, -1, structure_element2)
    update_pixels = region_to_close == 1
    image[update_pixels] = closed_image[update_pixels]
    return image


def extract_thin_vessels(image, eps_tn, r_th=3, scale_th1=1, scale_th2=1.5, s_tn=5, o_tn=100):
    # thin vessel background subtraction
    gf = guidedFilter(image, image, radius=r_th, eps=eps_tn)
    gf_image = cv2.subtract(image, gf)
    # hessian filtering
    hf_image = hbf.eigen_value_filter(gf_image, scale_th1, scale_th2)
    # Otsu's thresholding
    opt_threshold, thin_vessels = cv2.threshold(hf_image.astype('uint8'), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # pre process using area thresholding
    small_object_removed = morphology.remove_small_objects(thin_vessels.astype(bool), min_size=o_tn)
    # pre process using a modified closing procedure
    final_image = modified_closing(small_object_removed.astype('uint8'), s_tn)
    return final_image

###############################################################################


################################ IMAGE FUSION #################################

def fuse_images(thin_vessel_image, thick_vessel_image):
    fused_image = cv2.max(thin_vessel_image, thick_vessel_image)
    return fused_image

###############################################################################


############################ SATURATION NOISE MASK ############################

def extract_saturation_noise_mask(image, b_s, sig_s, s_s):
    # extract red channel 
    red_channel = image[:, :, 0]
    # Gaussian smoothing
    smoothed_image = cv2.GaussianBlur(red_channel, (b_s, b_s), sig_s)
    # otsu's thresholding
    opt_threshold, threshold_image = cv2.threshold(smoothed_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # closing of mask
    structure_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s_s, s_s))
    closed_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, structure_element)
    saturation_noise_mask = cv2.erode(closed_image, structure_element)
    return saturation_noise_mask

###############################################################################


def vessel_extraction_main(image, eps_ie, b_ie, w, n, kap, gam, eps_tn, sig_s, b_s, s_s, r_ie=3, sig_ie=16, s_tk=25,
                           scale_tk1=2, scale_tk2=3, o_tk=100, r_th=3, scale_th1=1, scale_th2=1.5, s_tn=5, o_tn=100):
    preprocessed_image = pre_process_image(image, eps_ie, b_ie, w, n, kap, gam, r_ie, sig_ie)
    thick_vessels = extract_thick_vessels(preprocessed_image, s_tk, scale_tk1, scale_tk2, o_tk)
    thin_vessels = extract_thin_vessels(preprocessed_image, eps_tn, r_th, scale_th1, scale_th2, s_tn, o_tn)
    fused_image = fuse_images(thin_vessels, thick_vessels)
    sat_mask = extract_saturation_noise_mask(image, b_s, sig_s, s_s)
    segmentation = cv2.multiply(fused_image, sat_mask)
    return segmentation

