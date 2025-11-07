# Tutorial Steps

I started this project while learning from "The Sound of AI" tutorial series on YouTube.  
This file gives a quick overview of the steps I followed to learn about neural networks and audio classification.

---

## Step 1 — Visualizing Audio Features
Notebook: `01_Visualizing_Audio_Features.ipynb`

In this step, I visualized simple audio signals and spectrograms.  
I learned how to read `.wav` files, use Short-Time Fourier Transform (STFT), and generate power spectrograms.  
This helped me understand what features like MFCC and Mel-spectrograms represent.

---

## Step 2 — Building MLPs with Keras
Notebook: `02_MLP_with_Keras.ipynb`

Here I trained my first deep learning model using TensorFlow and Keras.  
I created a simple MLP (Multi-Layer Perceptron) and tested it on small audio features to understand how neural networks classify data.

---

## Step 3 — CNN using MFCC Features
Notebook: `03_CNN_with_MFCC_Features.ipynb`

In this notebook, I used MFCC features from the UrbanSound8K dataset.  
I trained a CNN model and analyzed its accuracy, confusion matrix, and predictions.  
This was my first complete model pipeline for sound classification.

---

## Step 4 — CNN using Mel Features
Notebook: `04_CNN_with_Mel_Features.ipynb`

After MFCC, I tried Mel-spectrograms to compare their performance.  
The CNN structure was similar, but the Mel features gave different learning patterns.  
I also plotted training accuracy and loss to compare with the MFCC model.

---

## Step 5 — From Scratch Implementations (src/fundamentals/)
Scripts: `artificial_neuron.py`, `mlp_train_from_scratch.py`

These Python scripts are my low-level practice.  
Here I built a single artificial neuron and a full MLP from scratch using only NumPy, without TensorFlow or Keras.  
This helped me understand how forward propagation, backpropagation, and gradient descent actually work.

---

That’s the full learning path I followed from the math behind neurons to building full CNN models on real audio data. I guess beginners can follow the same approach and become more handy on these by doing couple of classification projects and then expand them.
