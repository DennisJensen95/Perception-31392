import torchvision
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(img):
  img = Image.fromarray(img) # Load the image
  img.thumbnail((128*4, 72*4), Image.ANTIALIAS)
  transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
  img = transform(img).to(device) # Apply the transform to the image
  pred = model([img]) # Pass the image to the model
  boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
  scores = pred[0]['scores'].detach().cpu().numpy()
  labels = pred[0]['labels'].cpu().numpy()
  return boxes, scores, labels

def object_detection_api(img, objects_to_detect, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    boxes, pred_cls, labels = get_prediction(img) # Get predictions
    # img = cv2.imread(img_path) # Read image with cv2
    # img = cv2.resize(img, (128*4, 72*4))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB

    for i in range(len(boxes)):

        for j in range(len(objects_to_detect)):
            if objects_to_detect[j] == COCO_INSTANCE_CATEGORY_NAMES[labels[i]] and pred_cls[i] > threshold:
                print(f'{COCO_INSTANCE_CATEGORY_NAMES[labels[i]]} - score: {pred_cls[i]}')
                cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
                center = tuple((np.asarray(boxes[i][0]) + np.asarray(boxes[i][1])) / 2)
                cv2.circle(img, center, radius=5, color=(255, 0, 0), thickness=5)
                # cv2.putText(img, pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th) # Write the prediction class

    return img
