from skimage import io, color, morphology, measure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import sobel
from skimage.morphology import erosion, dilation, disk, rectangle
from skimage.segmentation import flood
from scipy.ndimage import binary_fill_holes

def bg_image(image_path):    
    fig, ax = plt.subplots(1, 5, figsize=(12, 6))
    
    # Leer imagen
    image = img_as_float(io.imread(image_path))    
    
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    gray_image = color.rgb2gray(image) if len(image.shape) == 3 else image

    # Preprocesamiento
    Porcentaje = 0.4
    threshold = np.max(gray_image) * Porcentaje
    gray_image[gray_image <= threshold] = 0
    threshold = np.mean(gray_image)
    gray_image[gray_image <= threshold] = 0    
    
    ax[1].imshow(gray_image, cmap='gray')
    ax[1].set_title('double equalization')
    ax[1].axis('off')
        
    erosion_image = erosion(gray_image, rectangle(20, 20)) 
    dilation_image = dilation(erosion_image, disk(10)) 
    sobel_image = sobel(dilation_image)
    
    ax[2].imshow(sobel_image, cmap='gray')
    ax[2].set_title('Edges')
    ax[2].axis('off')

    # Detección de contornos y centroides
    binary_edges = sobel_image > 0.1
    labeled_image = measure.label(binary_edges)

    # Obtener los centroides de las regiones
    props = measure.regionprops(labeled_image)
    seed_points = [(int(prop.centroid[1]),int(prop.centroid[0])) for prop in props]  # Convertir centroides a enteros

    # Crear una máscara para la segmentación
    segmented_mask = np.zeros_like(gray_image, dtype=bool)
    threshold_regions = 0.8
    # Aplicar crecimiento de regiones desde cada semilla
    for seed in seed_points:
        if 0 <= seed[0] < gray_image.shape[0] and 0 <= seed[1] < gray_image.shape[1]:  # Asegurar que el punto está dentro de la imagen
            segmented_mask |= flood(gray_image, seed, tolerance=threshold_regions)  # Ajusta la tolerancia según necesites


    ax[3].imshow(segmented_mask, cmap='gray')
    ax[3].set_title('Segmented Regions')
    ax[3].axis('off')

    plt.show()
