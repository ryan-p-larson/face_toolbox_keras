import cv2
import numpy as np

class PreProcessor:
  @classmethod
  def scale_up(cls, image: np.ndarray, min_size: int = 1200) -> np.ndarray:
    if np.max(image.shape) < min_size:
      ratio = min_size / np.max(image.shape)
      gpu_mat = cv2.cuda_GpuMat()
      gpu_mat.upload(image)
      gpu_out = cv2.cuda.resize(gpu_mat, (0, 0), fx=ratio, fy=ratio)
      return gpu_out.download()
    return image

  @classmethod
  def scale_down(cls, image: np.ndarray, max_size: int = 768) -> np.ndarray:
    if np.max(image.shape) > max_size:
      ratio = max_size / np.max(image.shape)
      gpu_mat = cv2.cuda_GpuMat()
      gpu_mat.upload(image)
      gpu_out = cv2.cuda.resize(gpu_mat, (0,0), fx=ratio, fy=ratio)
      return gpu_out.download()
    return image

  @classmethod
  def colorize(cls, image: np.ndarray, code):
    gpu_mat = cv2.cuda_GpuMat()
    gpu_mat.upload(image)
    gpu_out = cv2.cuda.cvtColor(gpu_mat, code)
    return gpu_out.download()

  @classmethod
  def image_colors(cls, image: np.ndarray):
    height, width, channels = image.shape
    if (channels == 1):
      image   = cls.colorize(image, cv2.COLOR_GRAY2BGR)
    noiseless = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    gray      = cls.colorize(noiseless, cv2.COLOR_BGR2GRAY)
    ycrcb     = cls.colorize(noiseless, cv2.COLOR_BGR2YCrCb)
    return (image, noiseless, gray, ycrcb)

  @classmethod
  def image_threshold(cls, gray: np.ndarray, ycrcb: np.ndarray):
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
