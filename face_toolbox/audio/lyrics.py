import json
import pkg_resources


FRAMES_PER_SEC = 60
MS_PER_SECOND  = 1000
FRAMES_PER_MS  = MS_PER_SECOND / FRAMES_PER_SEC

class Lyrics:
  def __init__(self):
    lyric_file       = pkg_resources.resource_filename('face_toolbox', 'audio/lyrics.json')
    with open(lyric_file) as f:
      lyric_data     = json.load(f)

    self.lines       = lyric_data['lines']
    self.wordTimings = lyric_data['wordTimings']
    self.markers     = lyric_data['markers']
    self.N           = len(self.words)

  @property
  def words(self):
    return [w for l in self.lines for w in l.split()]

  @property
  def beat_times(self):
    return [t for l in self.wordTimings for t in l]

  @property
  def beat_times_ms(self):
    return [int(MS_PER_SECOND * t) for t in self.beat_times]

  @property
  def beat_frames(self):
    return [int(ms / FRAMES_PER_MS) for ms in self.beat_times_ms]

  def beat_delta_sec(self, idx: int):
    if (idx < 0 | idx > (self.N-1)):
      raise IndexError(idx)
    elif (idx == 0):
      return self.beat_times[idx]
    elif ((idx + 1) >= (self.N - 1)):
      return 38.0 - self.beat_times[self.N - 1]
    else:
      return self.beat_times[idx + 1] - self.beat_times[idx]

  def beat_delta_frames(self, idx: int):
    beat_sec_Δ = self.beat_delta_sec(idx)
    beat_ms_Δ  = beat_sec_Δ * MS_PER_SECOND
    frames_Δ   = int(beat_ms_Δ / FRAMES_PER_SEC)
    return frames_Δ
