<!-- # audio_processing
This is for a project based on the tutorial series from the youtube channel "Valerio Velardo-The Sound of AI" .

# Codes with UrbanSound
In this folder I made a classification project with the dataset of UrbanSound8k. In this project, audio classification is done based on Sound's different feature like mfcc's , tonnetz, mel spectrogram etc.
 -->


# Audio Processing for Deep Learning Tutorial: Learning Path from *The Sound of AI*

I started this project back in 2021, during the time of Covid. **“The Sound of AI”** YouTube tutorial series by Valerio Velardo made the things so easy for us.   

It became my first structured attempt to understand how sounds can be visualized, processed, and classified with neural networks.  

The repository now works as a complete **learning guide** for beginners who wish to explore the fundamentals of **audio machine learning** . This is covered from making a single artificial neuron to a working CNN model trained on the UrbanSound8K dataset.

---

## Overview

This project documents my progress while learning:
- How raw sound waves are transformed into numerical features  
- How neural networks learn to classify different sound patterns  
- How MFCC and Mel-spectrogram features affect model performance  

The materials are organized so that each step moves gradually from basic theory to hands-on implementation.

---

## Repository Structure

```
audio_processing/
├── notebooks/ # Step-by-step notebooks following the tutorial journey
├── src/ # Python source code (from-scratch + helper scripts)
├── docs/ # Additional notes and explanations
├── results/ # Training logs and figures (to be generated)
├── data/ # Sample .wav files
└── junk/ # Archived old or experimental scripts
```

---

## Learning Path

Each notebook builds on what came before.

| No. | Notebook | Description |
|----:|-----------|-------------|
| **1** | `01_Visualizing_Audio_Features.ipynb` | Introduces waveform and spectrogram visualization using STFT.  Demonstrates how frequency and time information appear in sound data. |
| **2** | `02_MLP_with_Keras.ipynb` | My first neural-network model using TensorFlow/Keras.  A basic MLP is trained and evaluated to understand dense-layer behavior. |
| **3** | `03_CNN_with_MFCC_Features.ipynb` | Uses MFCC features from UrbanSound8K to train a CNN.  Accuracy curves and predictions are plotted to observe learning progress. |
| **4** | `04_CNN_with_Mel_Features.ipynb` | Another CNN is trained, this time with Mel-spectrograms, and its results are compared with the MFCC model. |

---

## Core Source Files (`src/`)

| Path | Purpose |
|------|----------|
| `src/fundamentals/artificial_neuron.py` | Demonstrates a single neuron’s forward-propagation logic. |
| `src/fundamentals/mlp_train_from_scratch.py` | Implements an entire MLP from scratch with NumPy (backpropagation + gradient descent). |
| `src/features/feature_extraction_for_audio_cnn.py` | Handles MFCC feature extraction and saves them as HDF5 files. |
| `src/utils/read_plot_audio.py` | Utility for loading and plotting audio signals. |
| `src/utils/solving_overfitting.py` | Simple experiments for understanding dropout and regularization. |

These scripts complement the notebooks and can be run independently to review individual concepts.

---




---
## Documentation

Further details are stored inside the docs/ folder:

* docs/tutorial_steps.md
  --> complete learning sequence

* docs/notes_on_audio_features.md
 --> short explanations of MFCC, Mel, STFT

* docs/references.md
 --> datasets, tutorials, and tools used

## Acknowledgments

Parts of this project were inspired by:

* The Sound of AI — Valerio Velardo (YouTube Series)

* UrbanSound8K Dataset — https://urbansounddataset.weebly.com/urbansound8k.html

Special thanks go to the open-source communities behind **Librosa**, **TensorFlow**, and **Matplotlib** for providing the essential tools that made this exploration possible.