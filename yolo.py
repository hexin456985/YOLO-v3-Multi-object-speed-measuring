# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

import tracking

import time
from imutils.video import VideoStream
from imutils.video import FPS
import imutils

import pyrealsense2 as rs
import math

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/tiny_yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, color_frame, depth_frame, depth_intrin, dataFile):

        global whichMode
        global countTracking
        global maxTracking
        global nowOutBoxes
        global nowOutScore
        global nowOutClass
        global allTracker

        global lasttime
        global thistime
        global lastposition
        global thisposition
        global recordDistanceAndTime

        global lastGrayFrame
        global numberToCountVec

        print(whichMode)

        start = timer()

        # 计算上一帧和这一帧的区别框框
        judgeNone, allSmallBoxes, nowDealFrame = tracking.getMoveBox(lastGrayFrame, color_frame)
        if judgeNone == False:
            lastGrayFrame = nowDealFrame
            return image
        else:
            lastGrayFrame = nowDealFrame

        if(whichMode):
            lastposition = []
            thisposition = []

            if self.model_image_size != (None, None):
                assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
                assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
                boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
            else:
                new_image_size = (image.width - (image.width % 32),
                                  image.height - (image.height % 32))
                boxed_image = letterbox_image(image, new_image_size)

            image_data = np.array(boxed_image, dtype='float32')

            print(image_data.shape)
            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })

            # take the value to the global variables
            # nowOutBoxes = out_boxes
            # nowOutScore = out_scores
            # nowOutClass = out_classes
            nowOutBoxes = []
            nowOutScore = []
            nowOutClass = []

            # if box not interset with allSomeBoxes, then don't count
            for desCount in range(len(out_boxes)):
                if tracking.hasIntersect(allSmallBoxes, out_boxes[desCount]):
                    nowOutBoxes.append(out_boxes[desCount])
                    nowOutScore.append(out_scores[desCount])
                    nowOutClass.append(out_classes[desCount])
            out_boxes = nowOutBoxes
            out_scores = nowOutScore
            out_classes = nowOutClass

            print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

            # if find box number>0 then change to tracking mode
            if len(nowOutBoxes)>0:
                whichMode = False
                dataFile.write('Found {} boxes for {}\n'.format(len(out_boxes), 'img'))

            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                      size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 300

            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                print(label, (left, top), (right, bottom))
                dataFile.write(label+"\n")

                # save last position
                lasttime = time.time()
                x = int((left+right)/2)
                y = int((top+bottom)/2)
                dist_to_center = depth_frame.get_distance(x, y)
                depth_pixel = [x, y]
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dist_to_center)
                x1, y1, dist_to_center = depth_point
                lastposition.append([x1, y1, dist_to_center])

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])

                tempTracker = tracking.getNewTracker()
                tempTracker.init(np.asarray(image),(left, top, right-left, bottom-top))
                allTracker.append(tempTracker)

                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw

        else:
            countTracking = countTracking + 1

            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                      size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 300

            tempForCount = 0
            for i, c in reversed(list(enumerate(nowOutClass))):
                predicted_class = self.class_names[c]
                score = nowOutScore[i]

                (success, box) = allTracker[tempForCount].update(np.asarray(image))

                if success:
                    (x, y, w, h) = [int(v) for v in box]

                label = '{} {:.2f}'.format(predicted_class, score)

                left, top, width, height = box
                bottom = top + height
                right = left + width
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                print(label, (left, top), (right, bottom))

                thistime = time.time()
                x = int((left+right)/2)
                y = int((top+bottom)/2)
                dist_to_center = depth_frame.get_distance(x, y)
                depth_pixel = [x, y]
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dist_to_center)
                x1, y1, dist_to_center = depth_point

                if len(thisposition)-1<tempForCount:
                    thisposition.append([x1, y1, dist_to_center])
                else:
                    thisposition[tempForCount] = [x1, y1, dist_to_center]

                tempDistance = tracking.cal_Distance(lastposition[tempForCount][0],lastposition[tempForCount][1],lastposition[tempForCount][2],thisposition[tempForCount][0],thisposition[tempForCount][1],thisposition[tempForCount][2])
                tempTime = thistime - lasttime
                lastposition[tempForCount] = thisposition[tempForCount]

                recordDistanceAndTime.append([tempDistance, tempTime])

                label = label + ' speed{:.2f}'.format(tracking.calAvgSpeed(recordDistanceAndTime),numberToCountVec)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])

                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(255, 255, 255), font=font)
                dataFile.write(label+"\n")
                del draw

                tempForCount = tempForCount + 1

            if countTracking > maxTracking//len(nowOutBoxes)+1:
                whichMode = True
                countTracking = 0
                allTracker = []
                recordDistanceAndTime = []
                # 写入空行，分隔每组数据
                dataFile.write("\n")
                dataFile.write("\n")
                dataFile.write("\n")

        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()


# True means it is detecting mode
# False means it is tracking mode
whichMode = True
# tracking frame and max tracking frame
countTracking = 0
maxTracking = 20
# how many frame to calculate one vect
numberToCountVec = 5

# out_boxes, out_scores, out_classes
nowOutBoxes = []
nowOutScore = []
nowOutClass = []

# the number of now tracker
allTracker = []

# save the point thispoistion and lastposition
# save the time
lasttime = 0
thistime = 0
lastposition = []
thisposition = []
recordDistanceAndTime = []

# save last frame
lastGrayFrame = None

def detect_video(yolo, video_path="", output_path="", dataFile=None):
    import cv2
    # vid = cv2.VideoCapture(video_path)
    # vid = cv2.VideoCapture(0)
    # if not vid.isOpened():
    #     raise IOError("Couldn't open webcam or video")
    # video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    # video_fps       = vid.get(cv2.CAP_PROP_FPS)
    # video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #                     int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    out = None
    isOutput = True if output_path != "" else False
    if isOutput:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 15, (640,480))

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    while True:
        frames = pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        if not depth_frame or not color_frame:
            continue

        color_frame = np.asanyarray(color_frame.get_data())
        frame = color_frame

        # frame = imutils.resize(frame, width=640)
        image = Image.fromarray(frame)
        image = yolo.detect_image(image, color_frame, depth_frame, depth_intrin, dataFile)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(10, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(0, 0, 255), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pipeline.stop()
            break
    yolo.close_session()