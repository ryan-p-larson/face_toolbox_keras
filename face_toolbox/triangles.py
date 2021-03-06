import numpy as np
from shapely.geometry import Polygon
from skimage.draw import polygon as draw_polygon


def filter_triangle_points(points: np.ndarray, mask: np.ndarray):
  """Returns a Numpy array whose points fall within the boundaries of the valid mask area."""
  valid_points = [pt for pt in points if (mask[tuple(pt)] and mask[tuple(pt)] > 0)]
  return np.array(valid_points)

def randomize_triangle_points(points: np.ndarray, c: float) -> (np.ndarray, np.ndarray):
  num_points = int(np.where(points)[0].size * c)
  r, c       = np.nonzero(points)
  random_pts = np.zeros(r.shape) == 1
  random_pts[:num_points] = True
  np.random.shuffle(random_pts)

  selected_points = np.vstack([r[random_pts], c[random_pts]]).T
  filtered_points = np.vstack([r[random_pts == False], c[random_pts == False]]).T

  return (selected_points, filtered_points)

def compute_tri_pt_dist(pts: np.ndarray, origin: tuple = (0, 0)) -> float:
  """Return the Euclidean distance between a polygon's centroid and a reference point."""
  poly  = Polygon(pts)
  point = np.array(origin)
  return np.linalg.norm(poly.centroid.xy - point)

def sort_triangles(tris: np.ndarray):
  """Sort triangles from left to right, top to bottom."""
  key_by_dist = lambda p: compute_tri_pt_dist(p)
  sorted_tris = sorted([tri for tri in tris.copy()], key=key_by_dist)
  return sorted_tris

def split_triangles(tris: np.ndarray, amount: float):
  idx_split   = max(min(int(amount * len(tris)), len(tris)), 0)
  tris_before = tris[:idx_split]
  tris_after  = tris[idx_split:]
  return tris_before, tris_after

def sort_pts(pts: np.array):
  key_by_dist = lambda pt: np.linalg.norm(pt - np.array([0, 0]))
  sorted_pts  = sorted([pt for pt in pts.copy()], key=key_by_dist)
  return np.array(sorted_pts)

def tris_to_mask(image: np.ndarray, tris: np.ndarray):
  height, width = image.shape[:2]
  output = np.zeros((height, width), np.uint8)

  for tri in tris:
    rr, cc = draw_polygon(tri[:, 0], tri[:, 1], (height, width))
    output[rr, cc] = 1

  return output
