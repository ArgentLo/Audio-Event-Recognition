import os
import gc
import time
import shutil
import random
import warnings
import typing as tp
from pathlib import Path
from contextlib import contextmanager

import yaml
from joblib import delayed, Parallel

import cv2
import librosa
import audioread
import soundfile as sf

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import resnest.torch as resnest_torch

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions as ppe_extensions
from apex import amp  # mix-presicion training

import config as global_config
from dataset import SpectrogramDataset, INV_BIRD_CODE
from utils import *
from loss_func import BCEWithLogitsLoss_LabelSmooth, FocalLoss
import warnings
warnings.simplefilter("ignore")


def get_model(args: tp.Dict):
    model = getattr(resnest_torch, args["name"])(pretrained=args["params"]["pretrained"])
    del model.fc
    # use the same head as the baseline notebook.
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, args["params"]["n_classes"]))
    return model


def train_loop(manager, args, model, device, train_loader, 
               optimizer, scheduler, loss_func, fp16):
    """Run minibatch training loop"""
    while not manager.stop_trigger:
        model.train()
        progress_bar = tqdm(train_loader)
        for batch_idx, (data, target) in enumerate(progress_bar):
            with manager.run_iteration():
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_func(output, target)
                progress_bar.set_description(f'train/loss: {loss.item():.6f}')
                ppe.reporting.report({'train/loss': loss.item()})
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()                # in apex, loss.backward() becomes
                else:
                    loss.backward()                         # compute and sum gradients on params
                optimizer.step()
                scheduler.step()  # scheduler might be wrong position (but reportedly better perf.)


def eval_for_batch(args, model, device, data, target, 
                   loss_func, eval_func_dict={}):
    """
    Run evaliation for valid
    This function is applied to each batch of val loader.
    """
    model.eval()
    data, target = data.to(device), target.to(device)
    output = model(data)
    # Final result will be average of averages of the same size
    val_loss = loss_func(output, target).item()
    ppe.reporting.report({'val/loss': val_loss})
    
    for eval_name, eval_func in eval_func_dict.items():
        eval_value = eval_func(output, target).item()
        ppe.reporting.report({"val/{}".format(eval_name): eval_value})


def get_loaders_for_training(args_dataset: tp.Dict, args_loader: tp.Dict,
                             train_file_list: tp.List[str], val_file_list: tp.List[str]):
    
    # # make dataset
    train_dataset = SpectrogramDataset(train_file_list, **args_dataset)
    val_dataset = SpectrogramDataset(val_file_list, **args_dataset)
    # # make dataloader
    train_loader = data.DataLoader(train_dataset, **args_loader["train"])
    val_loader = data.DataLoader(val_dataset, **args_loader["val"])
    
    return train_loader, val_loader


def set_extensions(manager, args, model, device, test_loader, 
                   optimizer, loss_func, eval_func_dict={}):
    """set extensions for PPE"""
        
    my_extensions = [
        # # observe, report
        ppe_extensions.observe_lr(optimizer=optimizer),
        # ppe_extensions.ParameterStatistics(model, prefix='model'),
        # ppe_extensions.VariableStatisticsPlot(model),
        ppe_extensions.LogReport(),
        # ppe_extensions.PlotReport(['train/loss', 'val/loss'], 'epoch', filename='loss.png'),
        # ppe_extensions.PlotReport(['lr',], 'epoch', filename='lr.png'),
        ppe_extensions.PrintReport([
            'epoch', 'iteration', 'lr', 'train/loss', 'val/loss', "elapsed_time"]),
        # ppe_extensions.ProgressBar(update_interval=100),

        # evaluation
        (
            ppe_extensions.Evaluator(
                test_loader, model,
                eval_func=lambda data, target:
                    eval_for_batch(args, model, device, data, target, loss_func, eval_func_dict),
                progress_bar=False),
            (1, "epoch"),
        ),
        # save model snapshot.
        (
            ppe_extensions.snapshot(
                target=model, filename="snapshot_epoch_{.updater.epoch}.pth"),
            ppe.training.triggers.MinValueTrigger(key="val/loss", trigger=(1, 'epoch'))
        ),
    ]
           
    # set extensions to manager
    for ext in my_extensions:
        if isinstance(ext, tuple):
            manager.extend(ext[0], trigger=ext[1])
        else:
            manager.extend(ext)
    return manager


if __name__ == '__main__':

    # Hyper-parameter Settings
    settings = yaml.safe_load(global_config.SETTINGS_STR)
    # for k, v in settings.items():
    #     print("[{}]".format(k))
    #     print(v)

    ##############################################################
    ###############     Load dataset from PATH    ################
    ##############################################################

    tmp_list = []
    all_birds_dirs = Path(global_config.RESAMPLED_TRAIN_AUDIO_PATH)

    for ebird_d in all_birds_dirs.iterdir():
        if ebird_d.is_file():
            continue
        for wav_f in ebird_d.iterdir():
            tmp_list.append([ebird_d.name, wav_f.name, wav_f.as_posix()])
            
    print(f">>> Total training examples: {len(tmp_list)}")


    train_wav_path_exist = pd.DataFrame(
        tmp_list, columns=["ebird_code", "resampled_filename", "file_path"])

    train_all = pd.merge(
        global_config.train_csv, train_wav_path_exist, on=["ebird_code", "resampled_filename"], how="inner")


    ##############################################################
    #######   K-Fold split on each bird kind (ebird_code)  #######
    ##############################################################

    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=44)

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

    set_seed(settings["globals"]["seed"])
    device = torch.device(settings["globals"]["device"])
    output_dir = Path(settings["globals"]["output_dir"])

    # get loader
    train_loader, val_loader = get_loaders_for_training(settings["dataset"]["params"], settings["loader"], train_file_list, val_file_list)

    # get model
    model = get_model(settings["model"])
    model = model.to(device)

    # get optimizer
    optimizer = getattr(torch.optim, settings["optimizer"]["name"])(model.parameters(), **settings["optimizer"]["params"])
    scheduler = getattr(torch.optim.lr_scheduler, settings["scheduler"]["name"])(optimizer, **settings["scheduler"]["params"])
    loss_func = BCEWithLogitsLoss_LabelSmooth()  # getattr(nn, settings["loss"]["name"])(**settings["loss"]["params"])
    trigger = None

    if global_config.FP16:
        # APEX initialize -> FP16 training (half-precision)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=1)


    # PPE Manager
    manager = ppe.training.ExtensionsManager(
        model, optimizer, settings["globals"]["num_epochs"],
        iters_per_epoch=len(train_loader),
        stop_trigger=trigger,
        out_dir=output_dir
    )

    # set manager extensions
    manager = set_extensions(
        manager, settings, model, device,
        val_loader, optimizer, loss_func,
    )

    # Training Loop
    train_loop(manager, settings, model, device, train_loader,
            optimizer, scheduler, loss_func, global_config.FP16)
