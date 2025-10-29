import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

import librosa as lr

data_dir = 'audio_input/sample_test/'
audio_files = glob(data_dir + '/*.wav')
print(audio_files)
audio, sfreq = lr.load(audio_files[1])
time = np.arange(0,len(audio))/sfreq

print(time)
print("array length=",len(time))

#print(sfreq)

#plot audio
fig, ax = plt.subplots()
ax.plot(time, audio)
ax.set(xlabel='Time (s)', ylabel= 'Sound A')
plt.show()

