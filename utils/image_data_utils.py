import numpy as np 
import os 
import rasterio
from PIL import Image


def image2data(imgpath: str) -> tuple:
    if not os.path.isfile(imgpath):
        print(f'Duong dan anh {imgpath} khong ton tai')
        return None
    with rasterio.open(imgpath) as dataset:
        image_data = dataset.read()

    # Check the number of bands
        if image_data.shape[0] == 1:
            # The image is already in grayscale
            image_data = image_data
        else:
            # Convert to grayscale by averaging the bands
            image_data = np.mean(image_data, axis=0, keepdims=True)

    data = image_data.reshape((image_data.shape[1] * image_data.shape[2], image_data.shape[0]))
    data = data - np.min(data, axis=0)
    return data / np.max(data, axis=0), image_data.shape


def data2image(labels: np.ndarray, clusters: tuple, out_shape: tuple, output_path: str) -> None:
    segmented_image = labels.reshape(out_shape[1:])
    
    # Khai báo một bảng màu cho các cụm
    color_palette = np.array([
        [128, 128, 128],  # Gray (Urban/Concrete)
        [169, 169, 169],  # Dark Gray (Roads/Asphalt)
        [205, 133, 63],   # Peru (Buildings/Roofs)
        [240, 230, 140],  # Khaki (Dry Urban Areas)
        [34, 139, 34],    # Forest Green (Urban Vegetation)
        [0, 128, 0],      # Green (Parks/Gardens)
        [0, 0, 255],      # Blue (Deep Water/Rivers)
        [30, 144, 255],   # Dodger Blue (Shallow Water)
        [70, 130, 180],   # Steel Blue (Water Bodies)
        [210, 180, 140],  # Tan (Bare Soil/Earth)
        [244, 164, 96],   # Sandy Brown (Beaches/Sandbanks)
        [139, 69, 19],    # Saddle Brown (Bare Ground)
        [255, 255, 255],  # White (Clouds/Bright Surfaces)
        [105, 105, 105],  # Dim Gray (Shadows/Dark Areas)
        [0, 255, 0],      # Lime (Bright Vegetation)
    ], dtype=np.uint8)
    
    # Nếu số cụm lớn hơn số màu trong bảng màu, lặp lại bảng màu
    if len(clusters) > len(color_palette):
        # Lặp lại bảng màu cho đến khi đủ số cụm
        color_palette = np.tile(color_palette, (1 + len(clusters) // len(color_palette), 1))[:len(clusters)]
    
    # Tạo ảnh màu từ ảnh phân đoạn
    colored_segmented_image = color_palette[segmented_image]
    
    
    # Xử lý path đầu ra
    base, ext = os.path.splitext(output_path)
    index = 1
    while os.path.exists(output_path):
        output_path = f"{base}_{index}{ext}"
        index += 1
    
    Image.fromarray(colored_segmented_image).save(output_path, format='TIFF')
    print(f'Image saved to {output_path}')