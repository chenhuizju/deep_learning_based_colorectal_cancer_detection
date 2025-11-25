# deep_learning_based_colorectal_cancer_detection

The proposed pipeline for Colorectal cancer (CRC) detection in contrast-enhanced CT scans utilized the YOLOv11 network as the baseline architecture. A ResNet50 module was incorporated into the YOLOv11 backbone to enhance image feature extraction. Additionally, a scale-adaptive loss function, which introduces an adaptive coefficient and a scaling factor to adaptively measure the IoU and center point distance for improving box regression performance, was designed to further improve detection performance. 

The experiments were conducted on a Dell Alienware Aurora R16 Desktop with an Intel (R) Core (TM) 14th Gen i9 14900KF processor and NVIDIA GeForce RTX 4090 GPU. All Python code for the experiments was edited, compiled, and run in the Spyder integrated development environment (IDE). 

This code is based on Ultralytics package developed by Khanam et al. The weblink is https://docs.ultralytics.com/models/yolo11/

Here are the critical references:
Khanam, R., Hussain, M., 2024. Yolov11: An overview of the key architectural enhancements. arXiv preprint arXiv:2410.17725.
Khanam, R., Hussain, M., Hill, R., Allen, P., 2024. A comprehensive review of convolutional neural networks for defect detection in industrial applications. IEEE Access, 12, 94250-94295.

The CT data was converted from DICOM to 8-bit JPG format. The annotation file was in txt format (following YOLO series label requirements).


The steps to run the code were described as follows.
(1) use pip command or conda command to install the necessary packages/libraries listed in "requirements";


(2) dataset arrangement:
The data path is set as ***\\YOLOv11_CRC_X\\ultralytics\\cfg\\datasets\\jsv11_12.yaml
Note 1: "YOLOv11_CRC_X" is jsut the file name, you can change it to "YOLOv12_CRC_X" or anything else.
Note 2: the data structure must be set as follows.
path: ***\\YOLOv11_CRC_X\\ultralytics\\models\\yolo\\detect\\CRCCTdataset
train: images/train
val: images/test (or validate)
test: images/test
number of classes: 1
names:
   0: CRC


(3) when someone wants to train the model using his own dataset, please run the following command in the IPython Console:

from ultralytics import YOLO

model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training),load both yolo11s.yaml + yolo11s.pt

results = model.train(data="jsv11_12.yaml", epochs=100, imgsz=640) # imgsz=640 or imgsz=512, does not affect the results

Note 1: the test code is automatically run after every training epoch completed.
Note 2: for the CRC lesion detction in contrast-enhanced CT scans, We have trained and provided the wights file. Someone can use it directly for the same or similar tasks.  


(4) when someone wants to conduct the validation on the trained model, please run the following command in the IPython Console:

from ultralytics import YOLO

import pandas as pd

model = YOLO("***\\YOLOv11_CRC_X\\runs\\detect\\train\\weights\\last.pt")
or, model = YOLO("***\\YOLOv11_CRC_X\\runs\\detect\\train\\weights\\best.pt")

metrics = model.val(data="***\\YOLOv11_CRC_X\\ultralytics\\cfg\\datasets\\jsv11_12.yaml")

metrics_dict = metrics.results_dict

df = pd.DataFrame([metrics_dict])

df.to_csv("***\\YOLOv11_CRC_X\\runs\\detect\\val\\val_metrics_display.csv", index=False)


(5)when someone wants to conduct the predict (you will get bounding box detection results on the input CT scans) on the trained model, please run the following command in the IPython Console:

import yaml

from ultralytics import YOLO

import os

model = YOLO("***\\YOLOv11_CRC_X\\runs\\detect\\train\\weights\\last.pt")
or, model = YOLO("***\\YOLOv11_CRC_X\\runs\\detect\\train\\weights\\best.pt")

yaml_path = "***\\YOLOv11_CRC_X\\ultralytics\\cfg\\datasets\\jsv11_12.yaml"
with open(yaml_path, 'r', encoding='utf-8') as f:
    data_cfg = yaml.safe_load(f)


dataset_root = data_cfg.get("path", "") 
test_subpath = data_cfg.get("test", "")  
test_images_path = os.path.join(dataset_root, test_subpath)
print(f"Predictions will be performed on the test image path: {test_images_path}")

model.predict(
    source=test_images_path,
    save=True,
    save_txt=True,
    save_conf=True,
    conf=0.25,
    iou=0.50,
    imgsz=640, # 512
    device=0
)


Finally, if you utilized this code, please cite the article below.
[1] Chenhui Qiu, Sarah Miller, Barathi Subramanian, et al. A Deep Learning-Based Automated Pipeline for Colorectal Cancer Detection in Contrast-Enhanced CT Images. Computerized Medical Imaging and Graphics, XX (XX), pp: XX-XX.

 
