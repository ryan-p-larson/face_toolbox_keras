
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def get_track_beats(path: str, hop_length: int = 512) -> np.ndarray:
  y, sr = librosa.load(path)

  # Onset Envelope & Frames, times
  onset_env    = librosa.onset.onset_strength(y, sr, aggregate=np.median)
  onset_frames = librosa.onset.onset_detect(
      y, sr=sr, hop_length=hop_length, onset_envelope=onset_env, backtrack=True)
  onset_times   = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

  # Tempo and beats
  tempo, beat_times = librosa.beat.beat_track(
      y, sr=sr, onset_envelope=onset_env, trim=False,
      hop_length=hop_length, units='time')

  return beat_times

def get_track_clicks(beats, y, sr, hop_length: int = 512) -> np.ndarray:
  clicks = librosa.clicks(beats, sr=sr, length=len(y), hop_length=hop_length, click_duration=0.2)
  track_clicks = y + clicks
  return track_clicks

def show_beats(y, beats):
  fig, ax = plt.subplots(figsize=(18, 6))
  librosa.display.waveplot(y, alpha=0.5)
  plt.vlines(beat_times, -1, 1, color='r', alpha=0.75, linewidth=1.5)
  plt.ylim(-1, 1)
  plt.show()