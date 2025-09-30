from torchvision.io import decode_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

img = decode_image("/content/drive/MyDrive/dataset/cars.webp")

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.eval()

preprocess = weights.transforms()
batch = [preprocess(img)]

prediction = model(batch)[0]
labels = [weights.meta["categories"][i] for i in prediction["labels"]]
box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                          labels=labels,
                          colors="red",
                          width=4,
                          font_size=30,)

im = to_pil_image(box.detach())

plt.figure(figsize=(12, 10))
plt.imshow(im)
plt.axis('off')
plt.title("Object Detection Results")
plt.show()

