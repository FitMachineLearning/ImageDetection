import torch as th
import torchvision.transforms as T
import requests
from PIL import Image,ImageDraw,ImageFont

model = th.hub.load('facebookresearch/detr','detr_resnet101',pretrained=True)
model.eval()
model = model.cuda()
# url = 'https://s.abcnews.com/images/US/160825_vod_orig_historyofdogs_16x9_992.jpg'
url = 'https://colombiareports.com/wp-content/uploads/2019/07/f16s-1170x585.jpg'
# Image.open(requests.get(url, stream=True).raw).show()
img = Image.open(requests.get(url, stream=True).raw).resize((800,600))
# img.show()
transforms = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
img_tensor = transforms(img).unsqueeze(0).cuda()

print(img_tensor.shape)
with th.no_grad():
    output = model(img_tensor)

img2 = img.copy()
draw = ImageDraw.Draw(img2)

print(output['pred_logits'][0].shape)
for logits, box in zip(output['pred_logits'][0], output['pred_boxes'][0]):
    box_class = logits.argmax()
    if box_class >= len(CLASSES):
        continue
    label = CLASSES[box_class]
    box = box.cpu() * th.Tensor([800,600,800,600])
    x, y, w, h = box
    x0, x1 = x-w//2, x+w//2
    y0, y1 = y-h//2, y+h//2
    draw.rectangle([(x0,y0),(x1,y1)],width=2)
    draw.text((x0+4,y0+4),label)
    print(label , x , y , w , h)

img2.show()
