import nibabel as nib
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage import median_filter
from scipy import ndimage

def meanFilter(image, size):

    # Load image data
    image_data = image
    
    # Apply mean filter.
    filtered = uniform_filter(image_data, size=size)
    
    return filtered

def medianFilter(image, size):

    # Assign image
    image_data = image
    
    # Apply median filter
    filtered = median_filter(image_data, size=size)
    
    return filtered


def edgeDetection(image):
    print("hola")
    # Load image data
    image_data = image
    
    # Calculate the gradient in the x,y  and z directions
    sobel_x = ndimage.sobel(image_data, axis=0)
    sobel_y = ndimage.sobel(image_data, axis=1)
    sobel_z = ndimage.sobel(image, axis=2)
    
    # Calculate the magnitude of the gradient
    edges = np.sqrt(sobel_x**2 + sobel_y**2 + sobel_z**2)
    
    
    return edges