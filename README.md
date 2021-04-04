# Audio-Event Recognition with Deep Learning 

This is a Audio-Event Recognition framework based on `Deep CNN models`, specifically `ResNeSt` and `PANNs` model.

<p align="center">
    <img align="center" src="https://github.com/ArgentLo/Audio-Event-Recognition/blob/master/imgs/sound_0.png" width="380" height="195">
</p>

### Overall Framework

- [Sample Datasets](https://www.kaggle.com/c/birdsong-recognition/data?select=train_audio) ("Audio Event":  simply means audio-clip that has length >0s).

- **Audio Preprocessing**: 

  ➔ `Raw Audio` 

  ➔ `To Sound Wave`

  ➔ `Short-Time Fourier Transform` (STFT)

  ➔ `Mel Spectrogram`

- **Train CNN models** on `Mel Spectrogram` to classify each Audio Event.

  - **CNN Model details**:

    - Standard CNN model: ["ResNeSt: Split-Attention Networks"](https://arxiv.org/abs/2004.08955)

    - Audio-specialized CNN model: ["PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition"](https://arxiv.org/abs/1912.10211)


----

### Quick Start

- **Environment Requirement**

The code has been tested running under Python 3.7. The required packages are as follows:

```
pytorch == 1.3
torchlibrosa    # PyTorch librosa for Audio Analysis
catalyst        # PyTorch framework for Deep Learning Research
audioread       # cross-library Audio Decoding for Python
soundfile       # Read and Write Sound Files on Python
fastprogress
tqdm
```

- **Configuration**

All parameters, such as `batch_size`, `learning_rate`, `loss_func`, can be adjust to fix your need in `config.py`.

- **Training** 

```pyton
# for training ResNeSt
python train_resnest.py

# for training PANNs
python train_panns.py
```

----

### Audio Data Visualization

- **Sound Wave Visualization**
    - 
<p align="center">
    <img src="https://github.com/ArgentLo/Audio-Event-Recognition/blob/master/imgs/sound_1.png" width="668" height="420">
</p>

- **Mel Spectrogram Visualization**
    - A spectrogram is a visual way of representing the signal strength, or “loudness”, of a signal over time at various frequencies present in a particular waveform. Not only can one see whether there is more or less energy at, for example, 2 Hz vs 10 Hz, but one can also see how energy levels vary over time.
    - The Mel Spectrogram is a normal Spectrogram, but with a Mel Scale on the y axis.

<p align="center">
    <img src="https://github.com/ArgentLo/Audio-Event-Recognition/blob/master/imgs/mel_spec.png" width="680" height="420">
</p>









