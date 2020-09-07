import pandas as pd


# path to resume training
RESUME_WEIGHT = None # "./fold0/checkpoints/train.55.pth"

INIT_LR    = 1e-4
NUM_EPOCHS = 25
NUM_CYCLES = 1   # NUM_CYCLES for CosineAnnealingLR scheduler


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