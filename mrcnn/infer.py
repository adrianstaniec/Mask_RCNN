import os
import sys
import random
import math
import numpy as np
import skimage.io
import tensorflow as tf
import time
#import matplotlib
#import matplotlib.pyplot as plt
#os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

#%matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    #IMAGE_MIN_DIM = 416
    #IMAGE_MAX_DIM = 416

config = InferenceConfig()
config.display()

# Create model object in inference mode.
#model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
with tf.device("/gpu:7"):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,config=config)
#model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,config=config)
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

file_names = next(os.walk(IMAGE_DIR))[2]

image1 = skimage.io.imread('../images/12283150_12d37e6389_z.jpg')
image2 = image1

# Run detection
t1 = time.time()
results = model.detect([image1, image2], verbose=1)
print(time.time()-t1)
# Visualize results
r = results[0]
visualize.display_instances(image1, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])
print(r['rois'])
print(r['scores'])


# README:
# pip3 install --user cython
# pip3 install --user pycocotools
# cd /github/Mask_RCNN
# wget https://github.com/matterport/Mask_RCNN/releases/download/v1.0/mask_rcnn_coco.h5
# cd mrcnn/
# python3 infer.py
