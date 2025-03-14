import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from skimage import io, filters, color, exposure, morphology, measure, img_as_float
import matplotlib.pyplot as plt
from skimage.filters import rank
from skimage.morphology import flood, erosion, dilation, rectangle, disk
from skimage.util import img_as_float

def calculate_polygon_area(contour):
    x = contour[:, 1]
    y = contour[:, 0]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def find_meningioma_rect_crop(image_path, diameter_cm, porcentaje,tolerance_factor=1, width_ratio=0.5, height_ratio=0.7, max_attempts=5, tolerance_increment=1):
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
        threshold = np.max(cropped_image) * porcentaje 
        processed = cropped_image.copy()
        processed[processed <= threshold] = 0
        threshold = np.mean(processed)
        processed[processed <= threshold] = 0

        # Operaciones morfológicas
        processed = erosion(processed, rectangle(10,10))
        processed = dilation(processed, disk(5))

        # Binarización
        binary = processed > filters.threshold_otsu(processed)

        # Sobel
        binary = filters.sobel(binary)

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

    return gray_image, merge

def update_image(val=None):
    diameter = diameter_scale.get()
    porcentaje = porcentaje_scale.get() / 100.0
    
    original, processed = find_meningioma_rect_crop(image_path, diameter, porcentaje)
    
    ax1.clear()
    ax1.imshow(original, cmap='gray')
    ax1.set_title("Original")
    ax1.axis("off")
    
    ax2.clear()
    ax2.imshow(processed, cmap='gray')
    ax2.set_title(f"Procesada (D: {diameter} cm, P: {porcentaje:.2f})")
    ax2.axis("off")
    
    canvas.draw()

image_path = "brain-tumor-mri-dataset/Testing/meningioma/Te-me_0246.jpg"  # Reemplaza con la ruta de tu imagen

# Crear ventana principal
root = tk.Tk()
root.title("Segmentación de Meningioma")

# Crear marco para la imagen
frame = ttk.Frame(root)
frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Controles de parámetros
control_frame = ttk.Frame(root)
control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

style = ttk.Style()
style.configure("TScale", sliderlength=20, troughcolor="lightgray")

ttk.Label(control_frame, text="Diámetro (cm):").pack(side=tk.LEFT, padx=5)
diameter_scale = ttk.Scale(control_frame, from_=1, to=10, orient=tk.HORIZONTAL, command=update_image)
diameter_scale.pack(side=tk.LEFT, padx=10, pady=5)
diameter_scale.set(5)

ttk.Label(control_frame, text="Porcentaje:").pack(side=tk.LEFT, padx=5)
porcentaje_scale = ttk.Scale(control_frame, from_=1, to=100, orient=tk.HORIZONTAL, command=update_image)
porcentaje_scale.pack(side=tk.LEFT, padx=10, pady=5)
porcentaje_scale.set(50)

update_image()
root.mainloop()
