# Fuzzy Code 

## Introduction
This repository contains the implementation of Fuzzy C-Means (FCM) algorithm for clustering data and images. The FCM algorithm is implemented in Python using the Numpy library. The implementation is tested on the UCI dataset and the Anh-ve-tinh dataset.

## Folder Structure
```
├── config
│   ├── .yaml files
├── data
│   ├── csv
│   └── images
│       ├── Anh-da-pho
│       ├── Anh-ve-tinh
│       │   ├── Anh-da-pho
│       │   └── Anh-mau
├── fcm_script
│   ├── fcm_data.py
│   ├── fcm_image_folder.py
│   ├── fcm_image.py
│   ├── fcmp_data.py
│   └── fcmp_image.py
├── models
│   ├── fcm_parallel.py
│   ├── fcm.py
├── outputs
│   ├── images
│   ├── logs
├── README.md
├── utils
│   ├── image_data_utils.py
│   ├── load_dataset_UCI.py
│   ├── utils.py
│   └── validity.py
├── requirements.txt   
├── fcm_all_data.py
├── fcm_image_all.py
```

## How to use
- fcm_script folder contains the scripts to run the FCM algorithm on data and images. If you want to run the script inside this folder, please move it to the root folder. 
- The fcm_all_data.py and fcm_image_all.py are the scripts to run the FCM algorithm on all type of datasets and images. It will return the logs inside the outputs/logs folder and the images inside the outputs/images folder.

