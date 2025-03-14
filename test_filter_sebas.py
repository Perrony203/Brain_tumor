from skimage import io, filters, color, exposure, morphology, measure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import rank
from skimage.morphology import flood, erosion, dilation, disk, rectangle

def calculate_polygon_area(contour):
    x = contour[:, 1]
    y = contour[:, 0]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
def bg_image(image_path):    
    fig, ax = plt.subplots(1, 5, figsize=(12, 6))
    
    # Leer la imagen
    image = img_as_float(io.imread(image_path))    
    
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    if(len(image.shape) == 3 and image.shape[-1] == 3):
        # Convertir a escala de grises
        gray_image = color.rgb2gray(image)
    else:
        gray_image = image
    
    threshold = np.max(gray_image)*0.4
    
    gray_image[gray_image <= threshold] = 0
    threshold = np.mean(gray_image)
    gray_image[gray_image <= threshold] = 0    
        
    ax[1].imshow(gray_image, cmap='gray')
    ax[1].set_title('double equalization')
    ax[1].axis('off')
    
    # Binarizar la imagen
    binary_image = gray_image < filters.threshold_otsu(gray_image)

  
    ax[2].imshow(binary_image, cmap='gray')
    ax[2].set_title('binary')
    ax[2].axis('off')
    
    erosion_image = erosion(gray_image, rectangle(20,20)) 
    result_image = dilation(erosion_image, disk(10)) 

    # Resultado    
    ax[3].imshow(result_image, cmap='gray')
    ax[3].set_title('Result')
    ax[3].axis('off')

   
    plt.show()