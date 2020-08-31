import numpy as np
import cv2


def get_gradation_2d(start, stop, width, height, is_horizontal):
  if is_horizontal:
    return np.tile(np.linspace(start, stop, width), (height, 1))
  else:
    return np.tile(np.linspace(start, stop, height), (width, 1)).T

def get_gradation_3d(
  width: int,
  height: int,
  start_list: (int, int, int),
  stop_list: (int, int, int),
  is_horizontal_list: (bool, bool, bool) = (True, True, True)) -> np.ndarray:
  """
      Examples:
        get_gradation_3d(512, 256, (0, 0, 0), (255, 255, 255), (False, False, True))
  """
  result = np.zeros((height, width, len(start_list)), dtype=np.float)
  for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
    result[:, :, i] = get_gradation_2d(start, stop, width, height, is_horizontal)
  return result

def Laplacian_Pyramid_Blending_with_mask(
  A: np.ndarray,
  B: np.ndarray,
  m: np.ndarray,
  num_levels: int = 6):
  # assume mask is float32 [0,1]
  # generate Gaussian pyramid for A,B and mask
  GA = A.copy()
  GB = B.copy()
  GM = m.copy()
  gpA = [GA]
  gpB = [GB]
  gpM = [GM]
  for i in range(num_levels):
    GA = cv2.pyrDown(GA)
    GB = cv2.pyrDown(GB)
    GM = cv2.pyrDown(GM)
    gpA.append(np.float32(GA))
    gpB.append(np.float32(GB))
    gpM.append(np.float32(GM))

  # generate Laplacian Pyramids for A,B and masks
  lpA  = [gpA[num_levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
  lpB  = [gpB[num_levels-1]]
  gpMr = [gpM[num_levels-1]]
  for i in range(num_levels-1,0,-1):
    # Laplacian: subtarct upscaled version of lower level from current level
    # to get the high frequencies
    LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
    LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
    lpA.append(LA)
    lpB.append(LB)
    gpMr.append(gpM[i-1]) # also reverse the masks

  # Now blend images according to mask in each level
  LS = []
  for la,lb,gm in zip(lpA,lpB,gpMr):
    ls = la * gm + lb * (1.0 - gm)
    LS.append(ls)

  # now reconstruct
  ls_ = LS[0]
  for i in range(1,num_levels):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

  return ls_

def split_image(image: np.ndarray, direction: str = 'left') -> np.ndarray:
  height, width, channels = image.shape
  med_height, med_width   = height // 2, width // 2

  if direction == 'left':
    return image[0:height, 0:med_width]
  elif direction == 'top':
    return image[0:med_height, 0:width]
  elif direction == 'right':
    return image[0:height, med_width:width]
  else:
    return image[med:height, 0:width]
