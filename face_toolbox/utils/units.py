import numpy as np
from shapely.geometry import Polygon

_ORIGIN         = (0, 0)
_FRAMES_PER_SEC = 60
_FRAME_DURATION = 16.6667

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (  0,   0,   0)

def frame_idx_to_ts(idx: int):
  ms = round(idx * _FRAME_DURATION)
  return ms

def ts_to_frame_idx(ms: int):
  seconds     = round(ms / _FRAME_DURATION)
  return seconds

def compute_poly_org_dist(pts: np.ndarray) -> float:
  poly = Polygon(pts)
  point = np.array([0, 0])
  return np.linalg.norm(poly.centroid.xy - point)

def bgr_to_rgb(bgr: tuple):
  return (bgr[2], bgr[1], bgr[0])

def rgb_to_bgr(rgb):
  return (rgb[2], rgb[1], rgb[0])

def lighter(bgr: tuple, percent: float):
  '''assumes color is bgr between (0, 0, 0) and (255, 255, 255)'''
  rgb    = np.array(bgr_to_rgb(bgr))
  white  = np.array([255, 255, 255])
  vector = white - rgb
  lightened = tuple([int(i) for i in (rgb + vector * percent)])
  return rgb_to_bgr(lightened)
