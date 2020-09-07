import cv2
import numpy as np
from dlib import rectangle
from scipy.interpolate import interp1d
from face_toolbox.utils import visualize
import face_toolbox.triangles as triFuncs


def frame_bbox(image: np.ndarray, bbox: rectangle, thickness: int, color: tuple) -> np.ndarray:
  return visualize.draw_bbox(image, bbox.left(), bbox.top(), bbox.width(), bbox.height(), thickness, color)

def frame_face_landmarks(image: np.ndarray, landmarks: np.array, r: float, color: tuple) -> np.ndarray:
  return visualize.draw_landmarks(image, landmarks, r, color)

def frame_eyes_landmarks(image: np.ndarray, landmarks: np.array):
  def draw_polyline(img: np.ndarray, lndmrks: np.array, start: int, end: int, color: tuple, isClosed: bool = True):
    """Smoothly plots eyelid lines"""
    points              = lndmrks[start:end]
    distance            = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )))
    distance            = np.insert(distance, 0, 0) / distance[-1]
    interpolator        =  interp1d(distance, points, kind='cubic', axis=0)
    interpolated_points = interpolator(np.linspace(0, 1, 20))
    cv2.polylines(img, np.int32([interpolated_points]), isClosed, color, thickness=2)

  # Set up variables
  left   = landmarks[0][:, ::-1].astype(np.int32)
  right  = landmarks[1][:, ::-1].astype(np.int32)
  output = image.copy().astype(np.int32)

  # Eyelids
  draw_polyline(output, left,  0, 8, (255, 0, 0))
  draw_polyline(output, right, 0, 8, (255, 0, 0))

  # Pupils
  draw_polyline(output, left,  9, 16, (0, 0, 255))
  draw_polyline(output, right, 9, 16, (0, 0, 255))

  # Iris Center
  output = visualize.draw_landmarks(output, np.array([left[17]]),  5, (0, 255, 0))
  output = visualize.draw_landmarks(output, np.array([right[17]]), 5, (0, 255, 0))

  return output

def frame_eyes_triangle(image: np.ndarray, landmarks: np.array, thickness: int, color: tuple):
  # Set up variables
  left   = landmarks[0][:, ::-1].astype(np.int32)
  right  = landmarks[1][:, ::-1].astype(np.int32)
  output = image.copy().astype(np.int32)

  left_eye_center          = left[5]
  left_eye_x, left_eye_y   = left_eye_center[0], left_eye_center[1]
  right_eye_center         = right[13]
  right_eye_x, right_eye_y = right_eye_center[0], right_eye_center[1]
  eye_center_x_delta       = right_eye_x - left_eye_x
  eye_center_y_delta       = right_eye_y - left_eye_y

  # Iris Center
  output = visualize.draw_landmarks(output, np.array([left[17]]),  5, (0, 255, 0))
  output = visualize.draw_landmarks(output, np.array([right[17]]), 5, (0, 255, 0))

  # Offsets

  # Triangle lines
  output = cv2.line(output, tuple(left_eye_center), tuple(right_eye_center), 5, (0, 255, 0))
  output = cv2.line(output, tuple(left_eye_center), tuple(right_eye_center), 5, (0, 255, 0))
  output = cv2.line(output, tuple(left_eye_center), tuple(right_eye_center), 5, (0, 255, 0))

def frame_canny_split(image: np.ndarray, selected_pts: np.array, removed_pts: np.array, amount: float, select_color: tuple, remove_color: tuple):
  height, width              = image.shape[:2]
  removed_close, removed_far = triFuncs.split_triangles(removed_pts, amount)
  select_close,  select_far  = triFuncs.split_triangles(selected_pts, amount)
  output                     = visualize.draw_landmarks(np.zeros((height, width, 3), np.uint8), removed_far, r=1, color=remove_color)
  output                     = visualize.draw_landmarks(output, select_close, r=2, color=select_color)
  return output.astype(np.uint8)

def frame_outline_split(image: np.ndarray, tris: np.ndarray, amount: float,
  pt_color: tuple, tri_color: tuple, bg_color: tuple):
  height, width   = image.shape[:2]
  closer, further = triFuncs.split_triangles(tris, amount)
  further_pts     = np.array(further, np.int32).ravel().reshape((-1, 2))[:, ::-1]
  output          = visualize.draw_triangles_lines(image, closer, fg_color=tri_color, bg_color=bg_color)
  output          = visualize.draw_landmarks(output, further_pts, 1, pt_color)
  return output.astype(np.uint8)

def frame_fill_split(image: np.ndarray, tris: np.ndarray, amount: float, fg_color: tuple, bg_color: tuple):
  height, width   = image.shape[:2]
  closer, further = triFuncs.split_triangles(tris, amount)
  output          = visualize.draw_triangles_lines(image, tris, fg_color=fg_color, bg_color=bg_color)
  output          = visualize.draw_triangles_filled(image, closer, bg=output)
  return output.astype(np.uint8)
