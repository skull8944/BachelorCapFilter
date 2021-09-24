#!/usr/bin/python
from math import cos, sin, atan2
from cv2 import cv2
import numpy as np
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

def rotate(v, rotation):
  rot = np.array([[cos(rotation), -sin(rotation)], [sin(rotation), cos(rotation)]])
  return np.dot(rot, v)

def blit_image(src, dst, offset_x, offset_y):
  y1, y2 = np.clip(offset_y, 0, dst.shape[0]), np.clip(offset_y+src.shape[0], 0, dst.shape[0])
  x1, x2 = np.clip(offset_x, 0, dst.shape[1]), np.clip(offset_x+src.shape[1], 0, dst.shape[1])
  src_y1, src_y2 = 0-min(0, offset_y), 0-min(0, offset_y) + (y2-y1)
  src_x1, src_x2 = 0-min(0, offset_x), 0-min(0, offset_x) + (x2-x1)
  alpha_s = src[src_y1:src_y2, src_x1:src_x2, 3] / 255.0
  alpha_d = 1.0 - alpha_s
  for c in range(3):
    dst[y1:y2, x1:x2, c] = (alpha_s * src[src_y1:src_y2, src_x1:src_x2, c] + alpha_d * dst[y1:y2, x1:x2, c])

def draw_image(src, dst, x, y, rotation):
  (h, w) = src.shape[:2]
  bounds = np.array([
    [-w/2, -h/2], 
    [w/2, -h/2], 
    [w/2, h/2], 
    [-w/2, h/2]
  ])
  for i, v in enumerate(bounds):
    bounds[i] = rotate(v, rotation)+(x, y)
  x_min, x_max = min(np.concatenate(bounds[:, :1])), max(np.concatenate(bounds[:, :1]))
  y_min, y_max = min(np.concatenate(bounds[:, 1:])), max(np.concatenate(bounds[:, 1:]))
  M = cv2.getRotationMatrix2D((w/2, h/2), np.rad2deg(rotation), 1.0)
  src_rotated = cv2.warpAffine(src, M, (int(x_max-x_min), int(y_max-y_min)))
  (h, w) = src_rotated.shape[:2]
  blit_image(src_rotated, dst, int(x-w/2), int(y-h/2))

def face_rotation(landmarks):
  x = (landmarks.landmark[0].x - landmarks.landmark[1].x)
  y = (landmarks.landmark[0].y - landmarks.landmark[1].y)
  return atan2(x, y)

hat = cv2.imread("bachelor-cap.png", cv2.IMREAD_UNCHANGED)

# For webcam input:

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
        
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    if results.multi_face_landmarks:
      image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
      for face_landmarks in results.multi_face_landmarks:
        r = face_rotation(face_landmarks)
        d = ((abs(face_landmarks.landmark[332].x - face_landmarks.landmark[103].x)*image.shape[1])**2+(abs(face_landmarks.landmark[332].y - face_landmarks.landmark[103].y)*image.shape[0])**2)**0.5
        scale = d / hat.shape[1]*2
        scaled_hat = cv2.resize(hat, (int(hat.shape[1]*scale), int(hat.shape[0]*scale)))
        offset = rotate((0, -50), -r)
        draw_image(scaled_hat, image, face_landmarks.landmark[10].x*image.shape[1]+offset[0], face_landmarks.landmark[10].y*image.shape[0]+offset[1], r)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
