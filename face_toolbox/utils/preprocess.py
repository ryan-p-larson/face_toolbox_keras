import cv2
import numpy as np
from matplotlib import colors, cm

_HAS_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
_GPU_MAT = cv2.cuda_GpuMat() if (_HAS_CUDA) else None

def scale_up(image: np.ndarray, min_size: int = 1200) -> np.ndarray:
  if np.max(image.shape) < min_size:
    ratio = min_size / np.max(image.shape)
    if (_HAS_CUDA):
      _GPU_MAT.upload(image)
      gpu_out = cv2.cuda.resize(_GPU_MAT, (0, 0), fx=ratio, fy=ratio)
      return gpu_out.download()
    else:
      return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)
  return image

def scale_down(image: np.ndarray, max_size: int = 768) -> np.ndarray:
  if np.max(image.shape) > max_size:
    ratio = max_size / np.max(image.shape)
    if (_HAS_CUDA):
      _GPU_MAT.upload(image)
      gpu_out = cv2.cuda.resize(_GPU_MAT, (0, 0), fx=ratio, fy=ratio)
      return gpu_out.download()
    else:
      return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)
  return image

def add_padding_as_needed(image: np.ndarray, size: int, color: tuple) -> np.ndarray:
  height, width, channels = image.shape
  largest_dim   = max(height, width)

  resized_image = scale_down(image, size) if (largest_dim > size) else scale_up(image, size)
  resized_height, resized_width = resized_image.shape[:2]

  smallest_dim = min(resized_height, resized_width)
  left_right = resized_height > resized_width
  dimension_delta = size - smallest_dim

  if (left_right):
    return cv2.copyMakeBorder(resized_image, 0, 0, dimension_delta // 2, dimension_delta // 2, cv2.BORDER_CONSTANT, value=color)
  else:
    return cv2.copyMakeBorder(resized_image, dimension_delta // 2, dimension_delta // 2, 0, 0, cv2.BORDER_CONSTANT, value=color)


def colorize(image: np.ndarray, code):
  if (_HAS_CUDA):
    _GPU_MAT.upload(image)
    gpu_out = cv2.cuda.cvtColor(_GPU_MAT, code)
    return gpu_out.download()
  else:
    return cv2.cvtColor(image, code)

def warpAffine(image: np.ndarray, m: np.ndarray, size: (int, int)) -> np.ndarray:
  if (_HAS_CUDA):
    _GPU_MAT.upload(image)
    gpu_out = cv2.cuda.warpAffine(image, m, size)
    return gpu_out.download()
  else:
    return cv2.warpAffine(image, m, size))

def threshold_clahe(gray: np.ndarray, ycrcb: np.ndarray) -> (np.ndarray, float, float):
  if (_HAS_CUDA):
    gray_gpu = cv2.cuda_GpuMat()
    ycrcb_gpu = cv2.cuda_GpuMat()

    gray_gpu.upload(gray)
    ycrcb_gpu.upload(ycrcb)

    clahe = cv2.cuda.createCLAHE()
    normalized_gray_image = clahe.apply(gray_gpu, cv2.cuda_Stream.Null())
    high_thresh, threshold_image = cv2.threshold(ycrcb, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)# not supported

    low_thresh = 0.5 * high_thresh
    blurred_gray_image = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, -1, (3, 3), 16).apply(gray_gpu).download()
    sharp_gray_image = cv2.cuda.addWeighted(gray, 2.5, blurred_gray_image, -1, 0)

    return sharp_gray_image, low_thresh, high_thresh
  else:
    clahe = cv2.createCLAHE()
    normalized_gray_image  = clahe.apply(gray)
    high_thresh, thresh_im = cv2.threshold(ycbcr_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh             = 0.5 * high_thresh
    blurred_gray_image     = cv2.GaussianBlur(gray_image, (0, 0), 3)
    sharp_gray_image       = cv2.addWeighted(gray_image, 2.5, blurred_gray_image, -1, 0)
    return sharp_gray_image, low_thresh, high_thresh

def draw_segment_map(segments: np.ndarray):
  _SEGMENT_CMAP = [
    [ 0, '#ffffff'], # background
    [ 1, '#a1d99b'], # skin
    [ 2, '#41ab5d'], # eyebrow
    [ 3, '#41ab5d'], # eyebrow
    [ 4, '#238b45'], # eye
    [ 5, '#238b45'], # eye
    [ 6, '#6edf00'], # glasses
    [ 7, '#00fb75'], # ear
    [ 8, '#00fb75'], # ear
    [ 9, '#6edf00'], # earings
    [10, '#c7e9c0'], # nose
    [11, '#ffffcc'], # mouth
    [12, '#ffffcc'], # mouth
    [13, '#ffffcc'], # mouth
    [14, '#005718'], # neck
    [15, '#005718'], # neck_lower
    [16, '#decbe4'], # cloth
    [17, '#e5d8bd'], # hair
    [18, '#ffea00']  # hat
  ]
  for i in range(19, 256):
    _SEGMENT_CMAP.append([i, '#ffffff'])

  cmap_classes = colors.ListedColormap([seg[1] for seg in _SEGMENT_CMAP])
  rgba_data    = cm.ScalarMappable(cmap=cmap_classes).to_rgba(
      np.arange(0, 1.0, 1.0 / 256.0), bytes=True
  )
  rgba_data = rgba_data[:, 0:-1].reshape((256, 1, 3))

  # Convert to BGR (or RGB), uint8, for OpenCV.
  lut_classes = np.zeros((256, 1, 3), np.uint8)
  lut_classes[:, :, :] = rgba_data[:, :, ::-1]

  return cv2.applyColorMap(segments, lut_classes)

def segment_mask(
  image: np.ndarray,
  segments: np.ndarray,
  include: set = set([]),
  exclude: set = set([0, 16])
) -> np.ndarray:
  height, width = image.shape[:2]
  mask_output   = np.zeros((height, width, 1), dtype=np.uint8)

  for r in range(height):
    for c in range(width):
      included = ((len(include) == 0) or (segments[r][c] in include))
      excluded = segments[r][c] in exclude
      mask_output[r][c] = 1 if (included and not excluded) else 0

  # if (_HAS_CUDA):
  #   _GPU_MAT.upload(image)
  #   # gpu_out = cv2.cuda.bitwise_and(gpu_out, gpu_out, )
  masked_image = cv2.bitwise_and(image, image, mask=cv2.UMat(mask_output))
  return cv2.UMat.get(masked_image), mask_output