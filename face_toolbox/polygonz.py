import cv2
import dlib
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
from skimage.draw import polygon as draw_polygon, polygon2mask
from face_toolbox.segmenter import Segmenter


class Polygonz:
  def __init__(self, landmarks_path: str):
    self.detector     = dlib.get_frontal_face_detector()
    self.predictor    = dlib.shape_predictor(landmarks_path)
    self.cuda_enabled = cv2.cuda.getCudaEnabledDeviceCount() > 0
    self.mat          = cv2.cuda_GpuMat()# if (self.cuda_enabled) else None

  def upload(self, image: np.ndarray) -> None:
    if (self.cuda_enabled):
      self.mat.upload(image)

  def canny(self, gray: np.ndarray, a: int, b: int, c: float) -> np.ndarray:
    if (self.cuda_enabled):
      self.upload(gray)
      canny = cv2.cuda.createCannyEdgeDetector(a, b)
      edges = canny.detect(self.mat)
      edges = edges.download()
    else:
      edges = cv2.Canny(gray, a, b)

    num_points = int(np.where(edges)[0].size * c)
    r, c = np.nonzero(edges)
    random_points  = np.zeros(r.shape) == 1
    random_points[:num_points] = True
    np.random.shuffle(random_points)
    points = np.vstack([r[random_points], c[random_points]]).T

    return points

  def avg_masked(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    self.upload(image)
    if (self.cuda_enabled):
      mean, std_dev = cv2.cuda.meanStdDev(self.mat)
      return np.array(mean, np.uint8)
    else:
      return cv2.mean(image, mask)

  def create_triangles(self, mask: np.ndarray, gray: np.ndarray, a: int, b: int, c: float) -> np.ndarray:
    # canny edge detection
    points = self.canny(gray, a, b, c)

    # Detect faces if any
    detections = self.detector(gray, 1)
    shape      = self.predictor(gray, detections[0])
    # np.array([[shape.part(i).y, shape.part(i).x] for i in range(shape.num_parts)], np.int32)
    points = np.vstack([points,
        [[shape.part(i).y, shape.part(i).x] for i in range(shape.num_parts)]])


    # Create Del. triangulation and filter tris falling outside of mask
    delaunay = Delaunay(points, incremental=True)
    delaunay.close()
    triangles = delaunay.points[delaunay.simplices]
    filter_triangles = np.array([tri for tri in triangles
        if (mask[int(Polygon(tri).centroid.x), int(Polygon(tri).centroid.y)] > 0)])

    return filter_triangles

  def render_triangles(self, image: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    raise NotImplementedError('This method not implemented')

  def polygonize(self, image: np.ndarray, gray: np.ndarray, segments_mask: np.ndarray, a: int, b: int, c: float):
    triangles = self.create_triangles(segments_mask, gray, a, b, c)
    polygons  = self.render_triangles(image, triangles)
    return polygons


class PolygonzAvgNpRender(Polygonz):
  def __init__(self, landmarks_path: str):
    super().__init__(landmarks_path)

  def render_triangles(self, image: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    height, width, channels = image.shape
    output = np.full((height, width, 3), fill_value=255, dtype=np.uint8)
    for tri in triangles:
      rr, cc = draw_polygon(tri[:, 0], tri[:, 1], (height, width))
      color  = np.mean(image[rr, cc], axis=0)
      cv2.fillConvexPoly(output, tri[:, ::-1].astype(np.int32), color)
    return output

class PolygonzAvgCvRender(Polygonz):
  def __init__(self, landmarks_path: str):
    super().__init__(landmarks_path)

  def render_triangles(self, image: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    height, width, channels = image.shape
    output = np.full((height, width, 3), fill_value=255, dtype=np.uint8)
    self.upload(image)
    for tri in triangles:
      tri_mask = polygon2mask((height, width), tri)
      avg_color = self.avg_masked(image, tri_mask)
      cv2.fillConvexPoly(output, tri[:, ::-1].astype(np.int32), color)
    return output
