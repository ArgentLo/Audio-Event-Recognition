import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from contextlib import contextmanager
from pathlib import Path

from catalyst.dl import SupervisedRunner, State, CallbackOrder, Callback, CheckpointCallback
from fastprogress import progress_bar
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score


import config as global_config
from dataset import PANNsDataset, INV_BIRD_CODE
from model import PANNsCNN14Att
from model_layers import AttBlock, ConvBlock
from utils import *
from loss_func import PANNsLoss
import warnings
warnings.simplefilter("ignore")


def train_model():

    tmp_list = []
    all_birds_dirs = Path(global_config.RESAMPLED_TRAIN_AUDIO_PATH)

    for ebird_d in all_birds_dirs.iterdir():
        if ebird_d.is_file():
            continue
        for wav_f in ebird_d.iterdir():
            tmp_list.append([ebird_d.name, wav_f.name, wav_f.as_posix()])
            
    print(f">>> Total training examples: {len(tmp_list)}\n\n", tmp_list[:3])


    train_wav_path_exist = pd.DataFrame(
        tmp_list, columns=["ebird_code", "resampled_filename", "file_path"])

    train_all = pd.merge(
        global_config.train_csv, train_wav_path_exist, on=["ebird_code", "resampled_filename"], how="inner")


    ##############################################################
    #######   K-Fold split on each bird kind (ebird_code)  #######
    ##############################################################

    skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)

    train_all["fold"] = -1
    for fold_id, (train_index, val_index) in enumerate(skf.split(train_all, train_all["ebird_code"])):
        # df["fold"] == fold_id
        train_all.iloc[val_index, -1] = fold_id
        
    # check the propotion
    fold_proportion = pd.pivot_table(train_all, index="ebird_code", columns="fold", values="xc_id", aggfunc=len)
    print(f">>> Number of bird kinds: {fold_proportion.shape[0]} \n>>> Number of folds: {fold_proportion.shape[1]}")

    val_fold_num = 4

    train_file_list = train_all.query("fold != @val_fold_num")[["file_path", "ebird_code"]].values.tolist()
    val_file_list   = train_all.query("fold == @val_fold_num")[["file_path", "ebird_code"]].values.tolist()

    print(">>> Valid_Fold: [fold {}] train: {}, val: {}".format(val_fold_num, len(train_file_list), len(val_file_list)))



    ##############################################
    ###########       Model Setup      ###########
    ##############################################

    device = torch.device("cuda:0")

    # loaders
    loaders = {
        "train": data.DataLoader(PANNsDataset(train_file_list, None), 
                                batch_size=64, 
                                shuffle=True, 
                                num_workers=2, 
                                pin_memory=True, 
                                drop_last=True),
        
        "valid": data.DataLoader(PANNsDataset(val_file_list, None), 
                                batch_size=64, 
                                shuffle=False,
                                num_workers=2,
                                pin_memory=True,
                                drop_last=False)
    }


    # model
    global_config.model_config["classes_num"] = 527
    model = PANNsCNN14Att(**global_config.model_config)
    weights = torch.load(PRETRAIN_PANNS)

    # Load Pretrained Weight
    model.load_state_dict(weights["model"])
    model.att_block = AttBlock(2048, 264, activation='sigmoid')
    model.att_block.init_weights()

    ###################################################################
    # model.load_state_dict(torch.load("./fold0/checkpoints/train.14.pth")["model_state_dict"])
    ###################################################################


    model.to(device)
    print(f">>> Pretrained Model is loaded to {device}!")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=global_config.INIT_LR)

    # Scheduler
    NUM_EPOCHS = global_config.NUM_EPOCHS
    NUM_CYCLES = int(NUM_EPOCHS/(2*global_config.NUM_CYCLES))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_CYCLES)

    # Loss
    criterion = PANNsLoss().to(device)
    
    # Resume
    if global_config.RESUME_WEIGHT:
        # callbacks
        callbacks = [
            F1Callback(input_key="targets", output_key="logits", prefix="f1"),
            mAPCallback(input_key="targets", output_key="logits", prefix="mAP"),
            CheckpointCallback(save_n_best=5, resume=global_config.RESUME_WEIGHT)  # save 5 best models
        ]
    else:
        # callbacks
        callbacks = [
            F1Callback(input_key="targets", output_key="logits", prefix="f1"),
            mAPCallback(input_key="targets", output_key="logits", prefix="mAP"),
            CheckpointCallback(save_n_best=5)  # save 5 best models
        ]

    # Model Training
    runner = SupervisedRunner(
        device=device,
        input_key="waveform",
        input_target_key="targets"
    )

    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        verbose=True,
        logdir=f"fold0",
        callbacks=callbacks,
        main_metric="epoch_f1",  # metric to select the best ckpt
        minimize_metric=False
    )


if __name__ == '__main__':

    set_seed(444)
    train_model()