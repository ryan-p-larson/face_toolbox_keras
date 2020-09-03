import numpy as np
from colorsys import rgb_to_hls, hls_to_rgb

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

def compute_frame_delta(one, two):
  pass

def compute_ts_delta(one, two):
  pass


def compute_point_poly_dist(pts, point):
  poly = Polygon(pts)
  return np.distance(poly.centroid.xy, point)

def bgr_to_rgb(bgr: tuple):
  return (bgr[2], bgr[1], bgr[0])

def rgb_to_bgr(rgb):
  return (rgb[2], rgb[1], rgb[0])

def lighten_color(c: tuple, amount: float) -> tuple:
  rgb = bgr_to_rgb(c)
  print(f'rgb={rgb}')
  h, l, s = rgb_to_hls(rgb[0], rgb[1], rgb[2])
  l = max(min(l * amount, 1.0), 0.0)
  print(f'hls={h},{l},{s}')
  # lgt = hls_to_rgb(hls[0], max(0, min(1, amount * hls[1])), hls[2])
  lgt = hls_to_rgb(h, l, s)
  print(f'lgt={lgt}')
  return int(lgt[0] * 255), int(lgt[1] * 255), int(lgt[2] * 255)
