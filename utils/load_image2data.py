import numpy as np 
import os 

def image2data(imgpath: str) -> np.ndarray:
    import rasterio
    if not os.path.isfile(imgpath):
        print(f'Duong dan anh {imgpath} khong ton tai')
        return None
    with rasterio.open(imgpath) as dataset:
        image_data = dataset.read()

    data = image_data.reshape((image_data.shape[1] * image_data.shape[2], image_data.shape[0]))
    data = data - np.min(data, axis=0)
    return data / np.max(data, axis=0)