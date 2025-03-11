from skimage import io, filters, color, exposure, morphology, measure
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import rank
from skimage.morphology import flood


def highlight_meningiomas(image_path):
    # Leer la imagen
    image = io.imread(image_path)

    # Escala de grises
    gray_image = color.rgb2gray(image)

    # Suavizado
    smoothed_image = filters.gaussian(gray_image, sigma=1)

    # otsu local
    local_otsu = rank.otsu(smoothed_image, morphology.rectangle(3, 3))
    meningioma_mask = smoothed_image > local_otsu

    # Resultado
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(meningioma_mask, cmap='gray')
    ax[1].set_title('Meningioma Highlighted')
    ax[1].axis('off')
    plt.show()


def display_sobel_edges(image_path):
    #Leer la imagen
    image = io.imread(image_path)

    #Convertir a escala de grises
    gray_image = color.rgb2gray(image)

    # Sobel
    sobel_image = filters.sobel(gray_image)

    # Resultado
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(sobel_image, cmap='gray')
    ax[1].set_title('Sobel Edges')
    ax[1].axis('off')
    plt.show()


def display_equalized_image(image_path):
    # Leer la imagen
    image = io.imread(image_path)

    # Convertir a escala de grises
    gray_image = color.rgb2gray(image)

    # Equalizar el histograma
    equalized_image = exposure.equalize_hist(gray_image)

    # Resultado
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(equalized_image, cmap='gray')
    ax[1].set_title('Equalized Image')
    ax[1].axis('off')
    plt.show()


def preprocess_and_find_white_pixel(image_path):
    # Leer la imagen
    image = io.imread(image_path)

    # Convertir a escala de grises
    gray_image = color.rgb2gray(image)

    # Binarizar la imagen
    binary_image = gray_image > filters.threshold_otsu(gray_image)

    # Encontrar las coordenadas de los pixeles blancos
    white_pixel_coords = np.column_stack(np.where(binary_image))

    # Resultado
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(binary_image, cmap='gray')
    ax[1].set_title('Binary Image')
    ax[1].axis('off')
    plt.show()


    if white_pixel_coords.size > 0:
        return tuple(white_pixel_coords[0])
    else:
        return None


def region_growing(image_path, seed_point, threshold):
    #Leer la imagen
    image = io.imread(image_path)

    #Convertir a escala de grises
    gray_image = color.rgb2gray(image)

    #Aplicar un umbral para binarizar la imagen
    mask = flood(gray_image, seed_point, tolerance=threshold)

    # Resultado
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Region Grown')
    ax[1].axis('off')
    plt.show()

def calculate_polygon_area(contour):
    x = contour[:, 1]
    y = contour[:, 0]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def sobelpluswhite(image_path):
    # Leer la imagen
    image = io.imread(image_path)

    # Convertir a escala de grises
    gray_image = color.rgb2gray(image)

    # Binarizar la imagen
    binary_image = gray_image < filters.threshold_otsu(gray_image)

    # Sobel
    sobel_image = filters.sobel(binary_image)

    # Encontrar contornos
    contours = measure.find_contours(sobel_image, level=0.5)

    if not contours:
        print("No contours found in the image.")
        return

    # Encontrar el contorno más grande
    largest_contour = max(contours, key=calculate_polygon_area)

    # Encontrar el centroide del contorno más grande
    centroid = np.mean(largest_contour, axis=0)
    seed_point = tuple(map(int, centroid))

    # Crear una máscara usando el algoritmo de crecimiento de regiones
    mask = flood(sobel_image, seed_point, tolerance=0.1)

    merge= gray_image + mask

    # Resultado
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(merge, cmap='gray')
    ax[1].set_title('Result')
    ax[1].axis('off')
    plt.show()