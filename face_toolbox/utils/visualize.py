import cv2
import numpy as np
from skimage.draw import polygon as _polygon, polygon_perimeter
from matplotlib import pyplot as plt, colors, cm
from .units import lighter


_parsing_annos = [
    '0, background', '1, skin', '2, left eyebrow', '3, right eyebrow',
    '4, left eye', '5, right eye', '6, glasses', '7, left ear', '8, right ear', '9, earings',
    '10, nose', '11, mouth', '12, upper lip', '13, lower lip',
    '14, neck', '15, neck_l', '16, cloth', '17, hair', '18, hat'
]

def show_parsing_with_annos(data):
    # https://matplotlib.org/tutorials/colors/colormap-manipulation.html
    # https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
    fig, ax = plt.subplots(figsize=(8,8))

    #get discrete colormap
    cmap = plt.get_cmap('gist_ncar', len(_parsing_annos))
    new_colors = cmap(np.linspace(0, 1, len(_parsing_annos)))
    new_colors[0, :] = np.array([0, 0, 0, 1.])
    new_cmap = colors.ListedColormap(new_colors)

    # set limits .5 outside true range
    mat = ax.matshow(data, cmap=new_cmap, vmin=-0.5, vmax=18.5)

    #tell the colorbar to tick at integers
    cbar = fig.colorbar(mat, ticks=np.arange(0, len(_parsing_annos)))
    cbar.ax.set_yticklabels(_parsing_annos)
    plt.axis('off')
    fig.show()

def draw_bbox(image: np.ndarray, x: int, y: int, w: int, h: int, width: int, color: (int, int, int)):
    return cv2.rectangle(image.copy(), (x, y), (x+w, y+h), color, width, cv2.LINE_AA)

def draw_landmarks(image: np.ndarray, landmarks: np.ndarray, r: float, color: (int, int, int)):
    output = image.copy()
    for pt in landmarks:
        output = cv2.circle(output, tuple(pt), r, color, -1, cv2.LINE_AA)
    return output

def draw_triangles_fill_avg(image: np.ndarray, triangles: np.ndarray, color: (int, int, int)):
    height, width, channels = image.shape
    output = np.full((height, width, channels), color, np.int32)
    for tri in triangles:
        rr, cc = _polygon(tri[:, 0], tri[:, 1], (height, width))
        avg_color = np.mean(image[rr, cc], axis=0)
        cv2.fillConvexPoly(output, tri[:, ::-1].astype(np.int32), avg_color)
    return output

def draw_triangles_filled(
    image   : np.ndarray,
    tris    : np.ndarray,
    bg      : np.ndarray = None,
    bg_color: tuple      = None,
    amount  : float      = 1.0
):
    if (type(bg) != np.ndarray and bg_color == None): raise AssertionError('Background and color can not both be none')
    height, width = image.shape[:2]
    output = bg.astype(np.int32) if (bg_color == None) else np.full((height, width, 3), bg_color, np.int32)

    for tri in tris:
        rr, cc = _polygon(tri[:, 0], tri[:, 1], (height, width))
        average_color = np.mean(image[rr, cc], axis=0)
        average_color = tuple(average_color.astype(np.uint8).tolist())
        # lighter(tuple(average_color.tolist()), ...)
        try:
          lighter_color = lighter(average_color, amount) if (amount != 1.0) else average_color
        except Exception as e:
          print(average_color, average_color[0])
          print(amount)
          print(len(rr), len(cc))
          raise Exception(e)

        cv2.fillConvexPoly(output, tri[:, ::-1].astype(np.int32), lighter_color)

    return output

def draw_triangles_lines(
    image   : np.ndarray,
    tris    : np.ndarray,
    fg_color: tuple      = (0, 0, 0),
    bg_color: tuple      = None,
    bg      : np.ndarray = None,
    amount  : float      = 1.0
):
    if (type(bg) != np.ndarray and bg_color == None): raise AssertionError('Background and color can not both be none')
    height, width = image.shape[:2]
    output = bg if (bg_color == None) else np.full((height, width, 3), bg_color, np.int32)

    for tri in tris:
        rr, cc         = polygon_perimeter(tri[:, 0], tri[:, 1], (height, width))
        lighter_color  = lighter(fg_color, amount) if (amount != 1.0) else fg_color
        output[rr, cc] = lighter_color

    return output

def draw_triangles_outline(image: np.ndarray, triangles: np.ndarray, color: (int, int, int)):
    height, width, channels = image.shape
    output = np.full((height, width, channels), color, np.uint8)
    for tri in triangles:
        rr, cc         = polygon_perimeter(tri[:, 0], tri[:, 1], (height, width))
        output[rr, cc] = [255, 255, 255]
    return output

def draw_segment_map(segments: np.ndarray):
  _SEGMENT_CMAP  = [
    [ 0, '#ffffff'], # background
    [ 1, '#003f5c'], # skin
    [ 2, '#dd5182'], # eyebrow
    [ 3, '#dd5182'], # eyebrow
    [ 4, '#dd5182'], # eye
    [ 5, '#dd5182'], # eye
    [ 6, '#dd5182'], # glasses
    [ 7, '#955196'], # ear
    [ 8, '#955196'], # ear
    [ 9, '#955196'], # earings
    [10, '#955196'], # nose
    [11, '#F0AA99'], # mouth
    [12, '#F0AA99'], # mouth
    [13, '#F0AA99'], # mouth
    [14, '#955196'], # neck
    [15, '#955196'], # neck_lower
    [16, '#ff6e54'], # cloth
    [17, '#ff6e54'], # hair
    [18, '#ffa600']  # hat
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