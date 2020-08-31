import cv2
import numpy as np

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