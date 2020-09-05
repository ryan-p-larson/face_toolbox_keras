import numpy as np
from shapely.geometry import Polygon


def filter_triangle_points(points: np.ndarray, mask: np.ndarray):
  """Returns a Numpy array whose points fall within the boundaries of the valid mask area."""
  valid_points = [pt for pt in points if (mask[tuple(pt)] and mask[tuple(pt) > 0])]
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