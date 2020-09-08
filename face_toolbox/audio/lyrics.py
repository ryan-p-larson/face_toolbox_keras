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
    all_words = [w for l in lyric_data['lines'] for w in l.split()]
    all_times = [t for l in lyric_data['wordTimings'] for t in l]
    all_filt  = [wt for wt in zip(all_words, all_times)
      if (wt[0].lower().startswith('it') or wt[0].lower().endswith('1'))]

    self.words      = [wt[0] for wt in all_filt]
    self.beat_times = [wt[1] for wt in all_filt]

  @property
  def beat_times_ms(self):
    return [int(MS_PER_SECOND * t) for t in self.beat_times]

  @property
  def beat_frames(self):
    return [int(ms / FRAMES_PER_MS) for ms in self.beat_times_ms]

  def beat_delta_sec(self, idx: int):
    N = len(self.beat_times)
    if (idx < 0 | idx > (N - 1)):
      raise IndexError(idx)
    elif (idx == 0):
      return self.beat_times[idx]
    elif ((idx + 1) >= (N - 1)):
      return 38.0 - self.beat_times[N - 1]
    else:
      return self.beat_times[idx + 1] - self.beat_times[idx]

  def beat_delta_frames(self, idx: int):
    beat_sec_Δ = self.beat_delta_sec(idx)
    beat_ms_Δ  = beat_sec_Δ * MS_PER_SECOND
    frames_Δ   = int(beat_ms_Δ / FRAMES_PER_MS)
    return frames_Δ
