# Install detectron2

!python -m pip install pyyaml==5.1
import sys, os, distutils.core
!git clone 'https://github.com/facebookresearch/detectron2'
dist = distutils.core.run_setup("./detectron2/setup.py")
!python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])}
sys.path.insert(0, os.path.abspath('./detectron2'))

import torch, detectron2
!nvcc --version
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)



# Some basic setup

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog



# Run a pre-trained detectron2 model

!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg
im = cv2.imread("./input.jpg")
cv2_imshow(im)

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2_imshow(out.get_image()[:, :, ::-1])



# Train on a custom dataset

# prepare dataset
from google.colab import drive
drive.mount('/content/drive')

import shutil
import os

drive_train_path = '/content/drive/My Drive/pole'
drive_test11_path = '/content/drive/My Drive/pole_1-1'
drive_test12_path = '/content/drive/My Drive/pole_1-2'
drive_test21_path = '/content/drive/My Drive/pole_2-1'
drive_test22_path = '/content/drive/My Drive/pole_2-2'

local_train_path = '/content/pole'
local_test11_path = '/content/pole11'
local_test12_path = '/content/pole12'
local_test21_path = '/content/pole21'
local_test22_path = '/content/pole22'

for path in [local_train_path, local_test11_path, local_test12_path, local_test21_path, local_test22_path]:
    if os.path.exists(path):
        shutil.rmtree(path)

shutil.copytree(drive_train_path, local_train_path)
shutil.copytree(drive_test11_path, local_test11_path)
shutil.copytree(drive_test12_path, local_test12_path)
shutil.copytree(drive_test21_path, local_test21_path)
shutil.copytree(drive_test22_path, local_test22_path)

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import os
import json
import cv2
import numpy as np

def get_pole_dicts(img_dir):
    json_file = os.path.join(img_dir, "instances_default.json")
    with open(json_file, encoding='utf-8') as f:
        dataset_dicts = json.load(f)

    # Images list
    images = {img["id"]: img for img in dataset_dicts["images"]}

    # Annotations list
    annotations = dataset_dicts["annotations"]

    # Categories
    categories = {cat["id"]: cat["name"] for cat in dataset_dicts["categories"]}

    # Create the dataset dicts in Detectron2 format
    dataset_dicts = []
    for image_id, image in images.items():
        record = {}
        record["file_name"] = os.path.join(img_dir.rpartition('/annotations')[0] + "/images", image["file_name"])
        record["image_id"] = image_id
        record["height"] = image["height"]
        record["width"] = image["width"]

        objs = []
        for ann in annotations:
            if ann["image_id"] == image_id:
                obj = {
                    "bbox": ann["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": ann["segmentation"],
                    "category_id": ann["category_id"] - 1,  # Detectron2 uses 0-indexed category IDs
                    "iscrowd": ann.get("iscrowd", 0),
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


for d in ["train", "test"]:
    dataset_name = "pole_" + d
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(dataset_name)
        MetadataCatalog.remove(dataset_name)

categories = [
    {"id": 1, "name": "폴리머현수", "supercategory": ""},
    {"id": 2, "name": "접속개소", "supercategory": ""},
    {"id": 3, "name": "LA", "supercategory": ""},
    {"id": 4, "name": "TR", "supercategory": ""},
    {"id": 5, "name": "폴리머LP", "supercategory": ""}
]
thing_classes = [cat["name"] for cat in categories]

train_dicts = get_pole_dicts("/content/pole/annotations")
test11_dicts = get_pole_dicts("/content/pole11/annotations")
test12_dicts = get_pole_dicts("/content/pole12/annotations")
test21_dicts = get_pole_dicts("/content/pole21/annotations")
test22_dicts = get_pole_dicts("/content/pole22/annotations")
test_dicts = test11_dicts + test12_dicts + test21_dicts + test22_dicts

DatasetCatalog.register("pole_train", lambda: train_dicts)
MetadataCatalog.get("pole_train").set(thing_classes=thing_classes)

DatasetCatalog.register("pole_test", lambda: test_dicts)
MetadataCatalog.get("pole_test").set(thing_classes=thing_classes)

for d in random.sample(train_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("pole_train"), scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2_imshow(out.get_image()[:, :, ::-1])

for d in random.sample(test_dicts, 10):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("pole_test"), scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2_imshow(out.get_image()[:, :, ::-1])



# Train!
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("pole_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 10 
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1126 
cfg.SOLVER.STEPS = []  
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

%load_ext tensorboard
%tensorboard --logdir output



# Inference & evaluation using the trained model
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_pole_dicts("/content/pole/annotations")
for d in random.sample(dataset_dicts, 10):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  
    v = Visualizer(im[:, :, ::-1],
                   metadata=MetadataCatalog.get("pole_train"),
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("pole_test", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "pole_test")
print(inference_on_dataset(predictor.model, val_loader, evaluator))

