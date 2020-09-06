import cv2
import numpy as np
from skimage.morphology import remove_small_holes

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
    return cv2.warpAffine(image, m, size)

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

def segment_subset(mask: np.ndarray, include: set = set(), exclude: set = set([0, 16])):
  height, width = mask.shape[:2]
  output = np.zeros((height, width), np.uint8)
  for r in range(height):
    for c in range(width):
      included = ((len(include) == 0) or (mask[r][c] in include))
      excluded = mask[r][c] in exclude
      output[r][c] = 1 if (included and not excluded) else 0

  inpaint_mask   = remove_small_holes(output)
  inpaint_output = cv2.inpaint(output, ~inpaint_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
  return inpaint_output

def apply_mask(image: np.ndarray, mask: np.ndarray, color: tuple = (255, 255, 255)):
    height, width = image.shape[:2]
    background    = np.full((height, width, 3), color, np.uint8)

    # smooth the mask
    smoothed_mask    = cv2.GaussianBlur(mask, (7, 7), 0)
    image_foreground = cv2.bitwise_and(image, image, mask=smoothed_mask)
    image_background = cv2.bitwise_not(background, background, mask=smoothed_mask)
    combined_image   = cv2.add(image_background, image_foreground)

    return combined_image

def segment_mask(
  image: np.ndarray,
  segments: np.ndarray,
  include: set = set([]),
  exclude: set = set([0, 16])
) -> np.ndarray:
  height, width = image.shape[:2]
  mask_output   = np.zeros((height, width, 1), dtype=np.uint8)

  # Create mask from selected segments
  for r in range(height):
    for c in range(width):
      included = ((len(include) == 0) or (segments[r][c] in include))
      excluded = segments[r][c] in exclude
      mask_output[r][c] = 1 if (included and not excluded) else 0

  # Get the foreground by using the fresh mask
  # if (_HAS_CUDA):
  #   _GPU_MAT.upload(image)
  #   # gpu_out = cv2.cuda.bitwise_and(gpu_out, gpu_out, )
  masked_image = cv2.bitwise_and(image, image, mask=cv2.UMat(mask_output))
  return cv2.UMat.get(masked_image), mask_output

def grabcut_mask(image: np.ndarray, mask: np.ndarray, iter: int = 10):
  rough_output = cv2.bitwise_and(image, image, mask)
  fgModel      = np.zeros((1, 65), dtype='float')
  bgModel      = np.zeros((1, 65), dtype='float')

  gc_mask = mask.copy()
  gc_mask[gc_mask > 0]  = cv2.GC_PR_FGD
  gc_mask[gc_mask == 0] = cv2.GC_BGD
  gc_mask[gc_mask == 2] = cv2.GC_FGD
  gc_mask[gc_mask == 3] = cv2.GC_FGD
  gc_mask[gc_mask == 4] = cv2.GC_FGD
  gc_mask[gc_mask == 5] = cv2.GC_FGD
  gc_mask[gc_mask == 10] = cv2.GC_FGD
  gc_mask[gc_mask == 11] = cv2.GC_FGD
  gc_mask[gc_mask == 12] = cv2.GC_FGD
  gc_mask[gc_mask == 13] = cv2.GC_FGD
  gc_mask[gc_mask == 14] = cv2.GC_FGD

  (model_mask, bgModel, fgModel) = cv2.grabCut(
      image, gc_mask, None, bgModel, fgModel, iter, mode=cv2.GC_INIT_WITH_MASK)

  output_mask = np.where((model_mask == cv2.GC_BGD) | (model_mask == cv2.GC_PR_BGD), 0, 1)
  output_mask = (output_mask * 255).astype(np.uint8)
  output_soft = cv2.GaussianBlur(output_mask, (5, 5), 0)

  # apply a bitwise AND to the image using our mask generated by
  # GrabCut to generate our final output image
  output = cv2.bitwise_and(image, image, mask=output_mask)

  return output, output_mask