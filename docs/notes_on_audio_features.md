# Notes on Audio Features

When I started working with sound data, I realized that raw waveforms are hard for a model to understand.  
So I learned about the main ways to represent audio signals as features.

---

## 1. Spectrogram
A spectrogram shows **how the frequencies of a sound change over time**.  
I generated it using the Short-Time Fourier Transform (STFT).  
It helped me see which parts of a sound have higher or lower energy.

---

## 2. MFCC (Mel-Frequency Cepstral Coefficients)
MFCCs are one of the most common features in speech and sound recognition.  
They represent how humans perceive sound frequencies on a logarithmic (Mel) scale.  
I used 40 MFCCs per audio frame to train my CNN model in the MFCC notebook.

---

## 3. Mel-Spectrogram
A Mel-spectrogram looks similar to a normal spectrogram, but the frequency axis is mapped to the Mel scale.  
It captures perceptually important features of the sound.  
In my project, I trained a CNN using these Mel features and compared it with MFCC-based CNN performance.

---

## 4. Why Feature Extraction Matters
Instead of feeding the raw waveform into a neural network, feature extraction helps the model focus on the important information (frequency patterns) and reduces noise.  
It also makes training faster and more accurate.

