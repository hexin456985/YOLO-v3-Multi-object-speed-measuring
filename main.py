# tensorflow             1.15.4
# Keras                  2.1.5

import argparse
from yolo import YOLO, detect_video
from PIL import Image

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()

FLAGS = None
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
FLAGS = parser.parse_args()


# detect video(use your own camera)
with open("data.txt","w") as f:
    detect_video(YOLO(**vars(FLAGS)),"","result.avi",f)

# detect image
# detect_img(YOLO(**vars(FLAGS)))


