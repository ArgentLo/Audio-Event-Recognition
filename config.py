import pandas as pd


SETTINGS_STR = """
globals:
  seed: 1213
  device: cuda
  num_epochs: 100
  output_dir: ./fold0/
  use_fold: 0
  target_sr: 32000

dataset:
  name: SpectrogramDataset
  params:
    img_size: 224
    melspectrogram_parameters:
      n_mels: 128
      fmin: 20
      fmax: 16000
    
loader:
  train:
    batch_size: 128
    shuffle: True
    num_workers: 16
    pin_memory: True
    drop_last: True
  val:
    batch_size: 64
    shuffle: False
    num_workers: 2
    pin_memory: True
    drop_last: False

model:
  name: resnest50_fast_1s1x64d
  params:
    pretrained: True
    n_classes: 264

loss:
  name: BCEWithLogitsLoss
  params: {}

optimizer:
  name: Adam
  params:
    lr: 0.0008

scheduler:
  name: CosineAnnealingLR
  params:
    T_max: 20
"""


# path to resume training
RESUME_WEIGHT = None # "./fold0/checkpoints/train.55.pth"

INIT_LR    = 7.5e-4
NUM_EPOCHS = 100
NUM_CYCLES = 3   # NUM_CYCLES for CosineAnnealingLR scheduler
FP16       = True

model_config = {
    "sample_rate": 32000,
    "window_size": 1024,
    "hop_size": 320,
    "mel_bins": 64,
    "fmin": 50,
    "fmax": 14000,
    "classes_num": 264  # 527 in train
}

SR = 32000  # sampling rate

ROOT = "/home/argent/kaggle/Cornell_BirdSong_Reg/"
DATA_PATH = ROOT + "datasets/"

# train
RAW_TRAIN_AUDIO_PATH = DATA_PATH + "raw_train_audio/"
RESAMPLED_TRAIN_AUDIO_PATH = DATA_PATH + "resampled_train_audio/"
train_csv = pd.read_csv(RESAMPLED_TRAIN_AUDIO_PATH + "train_mod.csv")

# test
# TEST_AUDIO_DIR = DATA_PATH + "test_audio/"
TEST_AUDIO_DIR = DATA_PATH + "test_birdcall_check/test_audio/"
test_csv = pd.read_csv(DATA_PATH + "test_birdcall_check/test.csv")
