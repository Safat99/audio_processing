import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt

guitar_file = 'audio_input/sample_test/1.wav'
flute_file  = 'audio_input/sample_test/2.wav'

#print(guitar_file)
audio1, sfreq1 = librosa.load(guitar_file)

#ipd.Audio(guitar_file)


