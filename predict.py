from darkflow.net.build import TFNet
import cv2

options = {
    "model": "cfg/yolo-h2food.cfg", 
    "savedModelLoad": "saved_model",
    "threshold": 0.035, 
    "gpu": 1.0,
    "labels": "labels-h2food.txt",
}

img = cv2.imread("my_samples/13933.jpg")
tfnet = TFNet(options)
results = tfnet.return_predict(img)
print(results)
