import cv2
import numpy as np
from skimage.draw import polygon as _polygon
from matplotlib import pyplot as plt
from matplotlib import colors, cm


_parsing_annos = [
    '0, background', '1, skin', '2, left eyebrow', '3, right eyebrow',
    '4, left eye', '5, right eye', '6, glasses', '7, left ear', '8, right ear', '9, earings',
    '10, nose', '11, mouth', '12, upper lip', '13, lower lip',
    '14, neck', '15, neck_l', '16, cloth', '17, hair', '18, hat'
]

# https://matplotlib.org/tutorials/colors/colormap-manipulation.html
# https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
def show_parsing_with_annos(data):
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

def draw_segments(image: np.ndarray, segments: np.ndarray):
    pass

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

def draw_triangles_outline(image: np.ndarray, triangles: np.ndarray, width: float, color: (int, int, int)):
    height, width, channels = image.shape
    background = np.full((height, width, channels), color, np.int32)
    outlines   = cv2.polylines(background, triangles.astype(np.int32), True, (0, 0, 0), width, cv2.LINE_AA)
    return outlines

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