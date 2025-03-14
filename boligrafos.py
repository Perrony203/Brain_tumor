from skimage import data, color, util, filters, morphology, img_as_ubyte, io, exposure
import matplotlib.pyplot as plt
import numpy as np
import cv2

image_path = 'D:\\Universidad\\2025-1\\Computación gráfica\\Talleres\\Taller 1\\Imagenes\\OBJECTS.png'
image = io.imread(image_path, as_gray = True)
image = img_as_ubyte(image)

def og_color(og_image, segmented_image):
  new_image = segmented_image
  for i in range(len(segmented_image)):
    for j in range(len(segmented_image[i])):
      if segmented_image[i][j] == 255:
        new_image[i][j] = og_image[i][j]
  return new_image

def region_growing_prep(image, seeds, threshold):
  segmented = np.zeros_like(image, dtype=np.uint8)  # Máscara de la región

  for i in range(len(seeds)):
      height, width = image.shape
      visited = np.zeros_like(image, dtype=bool)  # Para evitar revisitar píxeles

      for j in range(len(segmented)):
        for k in range(len(segmented[j])):
          if segmented[j][k] == 255:
            visited[j][k] = True

      x, y = seeds[i]
      seed_value = image[y, x]
      stack = [(x, y)]  # Pila para expansión

      while stack:
          x, y = stack.pop()
          if visited[y, x]:  # Si ya se visitó, continuar
              continue

          visited[y, x] = True
          segmented[y, x] = 255  # Marcar píxel como parte de la región

          # Revisar píxeles vecinos (4-conectividad)
          for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
              nx, ny = x + dx, y + dy
              if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                  if abs(int(image[ny, nx]) - int(seed_value)) < threshold:
                      stack.append((nx, ny))

  return segmented

# Punto semilla
seeds = [(150, 20),(50, 110)]

# Umbral de contraste
threshold = 30

fig, axes = plt.subplots(1, 3, figsize=(10, 5))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Imagen Original',)
axes[0].axis('off')

# Ejecutar Region Growing
segmented_image = region_growing_prep(image, seeds, threshold)
s_i_b = np.count_nonzero(segmented_image == 255)
axes[1].imshow(segmented_image, cmap='gray')
axes[1].set_title(f'Imagen Segmentada\n(pixeles en 255 = {s_i_b})')
axes[1].axis('off')

recolor_image = og_color(image, segmented_image)
rc_i_b =np.count_nonzero(recolor_image == 255)
axes[2].imshow(recolor_image, cmap='gray')
axes[2].set_title(f'Imagen final\n(pixeles en 255 = {rc_i_b})')
axes[2].axis('off')

plt.show()
