from matplotlib import pyplot as plt
import numpy as np

def histogram(x):
  intensity = np.copy(x)
  color, frequency = np.unique(intensity, return_counts=True)
  histogram = np.full(256, 0,'uint64')
  histogram[color] = frequency
  return histogram

def globalEqualization(x):
  plt.imshow(x)  
  x = x.copy()
  hist = histogram(x)
  globalequ = np.round(255 * (np.cumsum(hist) / np.sum(hist))).astype('uint8')
  x = globalequ[x]
  # return x
  exit (0)