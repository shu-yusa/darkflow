from darkflow.net.build import TFNet
import tensorflow as tf
import cv2

options = {
    "model": "cfg/yolo-h2food.cfg", 
    "load": 1640, 
    "gpu": 1.0,
    "labels": "labels-h2food.txt",
}

export_dir = './saved_model'
tfnet = TFNet(options)

with tfnet.graph.as_default():
    tf.saved_model.simple_save(
        tfnet.sess,
        export_dir,
        inputs={'input': tfnet.inp},
        outputs={'output': tfnet.out})

writer = tf.summary.FileWriter("./graph", tfnet.sess.graph)
writer.close()
