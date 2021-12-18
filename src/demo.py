from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy.core.numeric import indices

import _init_paths


from opts import opts
from detectors.detector_factory import detector_factory

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

np.set_printoptions(threshold=sys.maxsize)

SIFT_params = dict( maxCorners = 100,
                      qualityLevel = 0.1,
                      minDistance = 7,
                      blockSize = 1)

# Parameters for lucas kanade optical flow
KLT_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors for tracking
color = np.random.randint(0, 255, (100, 3))

# Video framerate
FPS = 20

# Detections per second
DPS = 1/20

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)
  frame_idx = 0

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    ret, img_old = cam.read()
    while True:
      if frame_idx % (DPS * FPS) == 0:
        ret, img_new = cam.read()
        if not ret:
            print('No frames grabbed!')
            break

        cv2.imshow('input', img_new)
        result, debugger = detector.run(img_new)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, result[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit

        detection_gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
      elif (frame_idx - 1) % (DPS * FPS) == 0:
        ret, img_new = cam.read()
        if not ret:
            print('No frames grabbed!')
            break
          
        cv2.imshow('input', img_new)

        mask = np.zeros_like(img_new)
        mask, centers, indices = detector.create_detections_mask(debugger, mask, result['results'])
        
        current_gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
  
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(detection_gray, current_gray, centers, None, **KLT_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = centers[st==1]

        # Create a mask image for drawing purposes
        tracker = np.zeros_like(img_new)
        tracked = img_new.copy()
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            tracker = cv2.line(tracker, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            tracked = cv2.circle(tracked, (int(a), int(b)), 5, color[i].tolist(), -1)

        tracked = cv2.add(tracked, tracker)
        detector.custom_show_results(debugger, tracked, result['results'])

        # cv2.imshow('ctdet', tracked)
        if cv2.waitKey(1) == 27:
            return  # esc to quit

        # Now update the previous frame and previous points
        prev_gray = current_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

      else:
        _, img_new = cam.read()
        cv2.imshow('input', img_new)
        
        current_gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, p0, None, **KLT_params)
        
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        tracked = img_new.copy()
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            tracker = cv2.line(tracker, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            tracked = cv2.circle(tracked, (int(a), int(b)), 5, color[i].tolist(), -1)

        tracked = cv2.add(tracked, tracker)

        # This is prone to corrupting boxes, it is not perfect
        # detector.update_boxes(good_new, result['results'], indices)
        # detector.custom_show_results(debugger, tracked, result['results'])

        # Show tracked centers only
        cv2.imshow('ctdet', tracked)

        if cv2.waitKey(1) == 27:
            return  # esc to quit

        # Now update the previous frame and previous points
        prev_gray = current_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

      frame_idx += 1
      img_old = img_new
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    for (image_name) in image_names:
      result = detector.run(image_name)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, result[stat])
      print(time_str)
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
