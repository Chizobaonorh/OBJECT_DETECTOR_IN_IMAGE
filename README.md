# 🚗 Object Detection with Faster R-CNN (TorchVision)

This project demonstrates how to use **PyTorch’s torchvision library** with a pre-trained **Faster R-CNN model (ResNet-50 FPN backbone)** to perform object detection on an image. The model identifies objects in an image and draws bounding boxes around them with labels.

---

## 📌 Overview

* Loads an input image (`cars.webp`).
* Uses the **Faster R-CNN ResNet50 FPN v2** model pre-trained on COCO dataset.
* Preprocesses the image according to model requirements.
* Performs inference to detect objects in the image.
* Draws **bounding boxes and labels** on detected objects.
* Displays the results with `matplotlib`.

This setup is especially useful for tasks like vehicle detection, surveillance, and real-time scene understanding.

---

## ⚙️ Installation

Make sure you have the following dependencies installed:

```bash
pip install torch torchvision matplotlib
```

For GPU acceleration, ensure you have the right version of PyTorch + CUDA installed ([check compatibility here](https://pytorch.org/get-started/locally/)).

---

## 📂 Project Structure

```
.
├── cars.webp         # Input image
├── object_detection.py   # Main script
└── README.md         # Documentation
```

---

## 🧑‍💻 Code Walkthrough

```python
from torchvision.io import decode_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
```

* Import PyTorch vision utilities, the **Faster R-CNN model**, image preprocessing, and plotting tools.

```python
img = decode_image("/content/drive/MyDrive/dataset/cars.webp")
```

* Reads the input image into a tensor format.

```python
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.eval()
```

* Loads the pre-trained model with **COCO weights**.
* `box_score_thresh=0.9` ensures only high-confidence detections are shown.
* Sets model to **evaluation mode**.

```python
preprocess = weights.transforms()
batch = [preprocess(img)]
```

* Applies preprocessing transforms (resize, normalize, etc.) from the model’s weight configuration.
* Creates a batch with the processed image.

```python
prediction = model(batch)[0]
labels = [weights.meta["categories"][i] for i in prediction["labels"]]
```

* Runs inference on the batch.
* Retrieves predicted **labels** (like "car", "person", etc.) using COCO category metadata.

```python
box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                          labels=labels,
                          colors="red",
                          width=4,
                          font_size=30,)
```

* Draws bounding boxes on the detected objects with labels.

```python
im = to_pil_image(box.detach())
```

* Converts tensor with bounding boxes to a PIL image for visualization.

```python
plt.figure(figsize=(12, 10))
plt.imshow(im)
plt.axis('off')
plt.title("Object Detection Results")
plt.show()
```

* Displays the final image with detected objects highlighted.

---

## 📊 Example Output

When you run the script, you should see the input image with **red bounding boxes and labels** around detected objects.

---

## 🚀 Future Improvements

* Add support for **real-time object detection** via webcam.
* Experiment with **lower confidence thresholds** for more detections.
* Fine-tune the model on a **custom dataset** (e.g., only cars).
* Export results with bounding box coordinates for further analysis.

---

## 📜 References

* [Torchvision Object Detection Models](https://pytorch.org/vision/stable/models.html#object-detection)
* [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
* [COCO Dataset](https://cocodataset.org/)
