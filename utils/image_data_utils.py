import numpy as np 
import os 
import rasterio
import cv2
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
    # Reshape ảnh từ 1D về 2D, kích thước mong muốn 
    out_shape = (1024, 1024)
    segmented_image = labels.reshape(out_shape)

    # Tạo một ảnh với cùng số kênh như cụm
    colored_segmented_image = np.zeros(out_shape[0], out_shape[1], len(clusters), dtype=np.uint8 )

    # Gán màu tới các cụm   
    for i in range(len(clusters)):
        color = np.random.randint(0, 256, len(clusters))
        colored_segmented_image[segmented_image == i] = color

    im = Image.fromarray(colored_segmented_image)
    im.save(output_path, format='TIFF')
    print(f'Image saved to {output_path}')