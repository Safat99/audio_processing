#https://www.youtube.com/watch?v=Oa_d-zaUti8&ab_channel=ValerioVelardo-TheSoundofAI

import librosa,librosa.display
import matplotlib.pyplot as plt
import numpy as np


file = 'audio_input/sample_test/1.wav'

#waveform
signal, sr =librosa.load(file, sr=22050)
#signal 1 dimesional np array which contains  sr * T >> 22050 * audio_duration_in_sec
#librosa.display.waveplot(signal,sr=sr)
#plt.xlabel('Time')
#plt.ylabel('Amplitude')
#plt.show()

#fft -->spectrum to move from time domain to frequency domain
fft = np.fft.fft(signal)
#ekhane 1 dim np.array ashbe which has many values as total number of the waveform >> complex value ashbe 
magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]

#plt.plot(left_frequency, left_magnitude)
#plt.xlabel("Frequency")
#plt.ylabel("Magnitude")
#plt.show()


#stft ->spectrogram

n_fft = 2048
hop_length = 512

stft = librosa.core.stft(signal, hop_length = hop_length , n_fft = n_fft)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

#librosa.display.specshow(spectrogram, sr=sr, hop_length = hop_length)
#librosa.display.specshow(log_spectrogram, sr=sr, hop_length = hop_length)
#plt.xlabel("Time")
#plt.ylabel("Frequency")
#plt.colorbar()
#plt.show()

#MFCCs
MFCCs = librosa.feature.mfcc(signal, n_fft=n_fft , hop_length = hop_length, n_mfcc = 13)
librosa.display.specshow(MFCCs, sr=sr, hop_length = hop_length)
plt.xlabel("Time")
plt.ylabel("MFCCs")
plt.colorbar()
plt.show()




