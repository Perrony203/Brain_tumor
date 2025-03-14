from skimage import io, filters, color, exposure, morphology, measure
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import rank
from skimage.morphology import flood, erosion, dilation, rectangle, disk
from skimage.util import img_as_float


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
    # Leer la imagen
    image = io.imread(image_path)

    # Convertir a escala de grises
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
    # Leer la imagen
    image = io.imread(image_path)

    # Convertir a escala de grises
    gray_image = color.rgb2gray(image)

    # Aplicar un umbral para binarizar la imagen
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


def findMeningiomaByContours(image_path):
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

    merge = gray_image + mask

    # Resultado
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(merge, cmap='gray')
    ax[1].set_title('Result')
    ax[1].axis('off')
    plt.show()


def find_meningioma_by_contour_size(image_path, diameter_cm=10, tolerance_factor=1):
    # cargar la imagen
    image = img_as_float(io.imread(image_path))

    if (len(image.shape) == 3 and image.shape[-1] == 3):
        gray_image = color.rgb2gray(image)
    else:
        gray_image = image

    # calcular el radio en pixeles
    pixels_per_cm = 10.67  # Estimated conversion rate
    radius_pixels = (diameter_cm * pixels_per_cm) / 2
    reference_area = np.pi * (radius_pixels ** 2)

    # Definir el rango de tolerancia
    min_area = reference_area * (1 - tolerance_factor)
    max_area = reference_area * (1 + tolerance_factor)

    # Procesar la imagen
    threshold = np.max(gray_image) * 0.4
    gray_image[gray_image <= threshold] = 0
    threshold = np.mean(gray_image)
    gray_image[gray_image <= threshold] = 0

    erosion_image = erosion(gray_image, rectangle(20, 20))
    result_image = dilation(erosion_image, disk(10))

    # binarizar la imagen
    threshold = np.mean(result_image)
    binary_image = result_image > threshold

    # encontrar contornos
    contours = measure.find_contours(binary_image, level=0.5)

    if not contours:
        print("No contours found in the image.")
        return

    # Filtrar contornos por tamaño
    valid_contours = []
    for contour in contours:
        area = calculate_polygon_area(contour)
        if min_area <= area <= max_area:
            valid_contours.append((contour, area))

    if not valid_contours:
        print("No contours within the specified size range.")
        return

    # Find contour closest to reference area
    best_contour, _ = min(valid_contours, key=lambda x: abs(x[1] - reference_area))

    # encontrar el centroide del contorno más grande
    centroid = np.mean(best_contour, axis=0)
    seed_point = tuple(map(int, centroid))
    mask = flood(binary_image, seed_point, tolerance=0.1)

    merge = gray_image + mask

    # Resultado
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')
    ax[1].imshow(binary_image, cmap='gray')
    ax[1].set_title('Processed')
    ax[1].axis('off')
    ax[2].imshow(merge, cmap='gray')
    ax[2].set_title(f'Final Result (Target: {reference_area:.0f} px²)')
    ax[2].axis('off')
    plt.show()


def find_meningioma_rect_crop(image_path, diameter_cm, tolerance_factor=1, width_ratio=0.5, height_ratio=0.7, max_attempts=5, tolerance_increment=0.5):
    # Cargar y procesar imagen
    image = img_as_float(io.imread(image_path))

    if(len(image.shape) == 3 and image.shape[-1] == 3):
        gray_image = color.rgb2gray(image)
    else:
        gray_image = image

    # Calcular dimensiones del recorte rectangular
    height, width = gray_image.shape
    crop_width = int(width * width_ratio)
    crop_height = int(height * height_ratio)

    # Calcular coordenadas del recorte
    start_y = (height - crop_height) // 2
    start_x = (width - crop_width) // 2

    # Recortar imagen
    cropped_image = gray_image[start_y:start_y+crop_height, start_x:start_x+crop_width]

    # Calcular área de referencia
    pixels_per_cm = 10.67
    radius_pixels = (diameter_cm * pixels_per_cm) / 2
    reference_area = np.pi * (radius_pixels ** 2)

    def try_find_contours(current_tolerance, attempt=1):
        if attempt > max_attempts:
            print(f"No se encontraron contornos después de {max_attempts} intentos.")
            return None

        # Definir rango de área con tolerancia actual
        min_area = reference_area * (1 - current_tolerance)
        max_area = reference_area * (1 + current_tolerance)

        # Procesar imagen
        threshold = np.max(cropped_image) * 0.4
        processed = cropped_image.copy()
        processed[processed <= threshold] = 0
        threshold = np.mean(processed)
        processed[processed <= threshold] = 0

        # Operaciones morfológicas
        processed = erosion(processed, rectangle(10,10))
        processed = dilation(processed, disk(5))

        # Binarización
        binary = processed > filters.threshold_otsu(processed)

        # Encontrar contornos
        contours = measure.find_contours(binary, level=0.5)

        if not contours:
            print(f"Intento {attempt}: No se encontraron contornos. Aumentando tolerancia...")
            return try_find_contours(current_tolerance + tolerance_increment, attempt + 1)

        # Filtrar contornos por área
        valid_contours = []
        for contour in contours:
            area = calculate_polygon_area(contour)
            if min_area <= area <= max_area:
                valid_contours.append((contour, area))

        if not valid_contours:
            print(f"Intento {attempt}: No hay contornos válidos. Aumentando tolerancia...")
            return try_find_contours(current_tolerance + tolerance_increment, attempt + 1)

        # Encontrar el mejor contorno
        best_contour, _ = min(valid_contours, key=lambda x: abs(x[1] - reference_area))
        return best_contour, binary, processed

    # Intentar encontrar contornos con recursividad
    result = try_find_contours(tolerance_factor)
    if result is None:
        return

    best_contour, binary, processed = result

    # Aplicar relleno por inundación
    centroid = np.mean(best_contour, axis=0)
    seed_point = tuple(map(int, centroid))
    mask = flood(binary, seed_point, tolerance=0.1)

    # Crear máscara de tamaño completo
    full_mask = np.zeros_like(gray_image)
    full_mask[start_y:start_y+crop_height, start_x:start_x+crop_width] = mask

    merge = gray_image + full_mask

    # Mostrar resultados
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')
    ax[1].imshow(cropped_image, cmap='gray')
    ax[1].set_title('Recortada')
    ax[1].axis('off')
    ax[2].imshow(processed, cmap='gray')
    ax[2].set_title('Procesada')
    ax[2].axis('off')
    ax[3].imshow(binary, cmap='gray')
    ax[3].set_title('Binarizada')
    ax[3].axis('off')
    ax[4].imshow(merge, cmap='gray')
    ax[4].set_title(f'Resultado Final (Objetivo: {reference_area:.0f} px²)')
    ax[4].axis('off')
    plt.show()