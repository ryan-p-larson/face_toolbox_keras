from typing import List
import cv2
import numpy as np
from .models.parser.face_parser import FaceParser
# from .preprocessor import PreProcessor

class Segmenter:
  def __init__(self, model: FaceParser):
    self.model = model

  @classmethod
  def mask(
    cls,
    image: np.ndarray,
    segments: np.ndarray,
    include: List[int] = [],
    exclude: List[int] = [0]):
    include, exclude = set(include), set(exclude)
    height, width, _ = image.shape
    mask_out = np.zeros((height, width, 1), dtype=np.uint8)

    for r in range(height):
      for c in range(width):
        included = ((len(include) == 0) or (segments[r][c] in include))
        excluded = segments[r][c] in exclude
        mask_out[r][c] = 1 if (included and not excluded) else 0

    masked_image = cv2.bitwise_and(image, image, mask=cv2.UMat(mask_out))
    return cv2.UMat.get(masked_image), mask_out

  def segment(self, image: np.ndarray) -> (np.ndarray, np.ndarray):
    # First, resize the image if it's too small
    height_og, width_og = image.shape[:2]
    # resized_image = PreProcessor.scale_down(image, 1200)

    # Second, segment images according to model
    faces = self.model.parse_face(image) #resized_image
    segs  = faces[0]

    # Construct mask by filtering/normalizing classes
    _, mask = self.mask(image, segs, exclude=[0, 16])

    # Finally, resize the image to the original image passed in
    # resized_mask = PreProcessor.scale_up(mask, max(height_og, width_og))
    # resized_segs = PreProcessor.scale_up(segs, max(height_og, width_og))
    # return resized_mask, resized_segs
    return mask, segs