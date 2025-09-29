import os, urllib.request
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

# Paths for YOLO files
yolo_dir = os.path.expanduser("~/.cvlib/object_detection/yolo")
os.makedirs(yolo_dir, exist_ok=True)

files = {
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
    "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "yolov3.txt": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
}
for fname, url in files.items():
    fpath = os.path.join(yolo_dir, fname)
    if not os.path.exists(fpath) or os.path.getsize(fpath) < 1000:
        print(f"Downloading {fname}...")
        urllib.request.urlretrieve(url, fpath)

# Test with an image # replace with your image path
box, label, count = cv.detect_common_objects(img, model="yolov3")
output = draw_bbox(img, box, label, count)


img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.axis("off")
plt.imshow(img)
plt.show()

cv2.imwrite("output.jpg", output)
print("Objects:", label)