import numpy as np 
import os 
import rasterio
from PIL import Image


def image2data(imgpath: str) -> np.ndarray:
    if not os.path.isfile(imgpath):
        print(f'Duong dan anh {imgpath} khong ton tai')
        return None
    with rasterio.open(imgpath) as dataset:
        image_data = dataset.read()

    data = image_data.reshape((image_data.shape[1] * image_data.shape[2], image_data.shape[0]))
    data = data - np.min(data, axis=0)
    return data / np.max(data, axis=0)


def data2image(labels: np.ndarray, clusters: tuple, output_path: str) -> None:
    out_shape = (1024, 1024)
    segmented_image = labels.reshape(out_shape)
    
    # Khai báo một bảng màu cho các cụm
    color_palette = np.array([
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [128, 0, 0],    # Maroon
        [0, 128, 0],    # Dark Green
        [0, 0, 128],    # Navy
        [128, 128, 0],  # Olive
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