# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch

import _init_paths
from config import cfg
from config import update_config

import models
import torchinfo
import pathlib


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    parser.add_argument("--checkpoint", 
                        type=str,
                        default=None, 
                        help="checkpoint file to load.")
    parser.add_argument("--load_messy_checkpoint", action="store_true", default=False, help="if the official checkpoints are to be used. with this flag, we will try to load with a heuriritic as below to fix the checkpoint.")

    args = parser.parse_args()
    return args


def load_messy_checkpoint(model, checkpoint_path: str):
    """
    loads the checkpoint from the official github repo weights
    
    Note: 
    1. please make sure you are passing the correct weigths corresponding to the
    model cfg you are using. 
    2. this is experimental logic and you can try to find the mapping yourself between the missing keys and the unexpected keys 
    
    """
    
    # load the checkpoint 
    checkpoint = torch.load(checkpoint_path)
    
    # generate a copy. 
    state_dict = checkpoint['state_dict']
    state_dict_copy = state_dict.copy()
    state_dict_copy.clear() 
    
    # summary path 
    summary_path = str(pathlib.Path(checkpoint_path).with_suffix(".txt"))
    print(f"check summary path : {summary_path}")
    
    # new checkpoint path
    new_filename = f"fixed_{pathlib.Path(checkpoint_path).stem}"
    new_ckpt_path = str(pathlib.Path(checkpoint_path).with_stem(new_filename))
    
    print(f"fixed checkpoinnt will be saved to: {new_ckpt_path}")
    
    # make the key names to be matching 
    for k in list(state_dict.keys()):
        
        if k.find("bn") != -1:
            print(f"key : {k}, replacing bn with norm")
            new_key = k.replace("bn", "norm")
        else:
            new_key = k 
                    
        if new_key.startswith("backbone."):
            print(f"key : {new_key}, replacing backbone with empty")
            new_key = new_key.replace("backbone.", "")
            
        if new_key.startswith("keypoint_head."):
            print(f"key : {new_key}, replacing keypoint_head with empty")
            new_key = new_key.replace("keypoint_head.", "")
            
        state_dict_copy[new_key] = state_dict[k]
        
        
    try: 
        info = model.load_state_dict(state_dict_copy, strict=False)
        
        # print(f"missing keys : {info.missing_keys}")
        # print(f"\n\n")
        # print(f"unexpected keys : {info.unexpected_keys}")
    
        with open(summary_path, "w") as f:
            f.write(f"missing keys: {str(info.missing_keys)}")
            f.write("\n\n")
            f.write(f"unexpected keys: {str(info.unexpected_keys)}")
                
        if info.missing_keys == [] and info.unexpected_keys == []:
            print(f"info = {info}")
            torch.save(state_dict_copy, new_ckpt_path)
            print(f"saved to {new_ckpt_path}")
        
        
    except Exception as e: 
        print(e)
    



def main():
    args = parse_args()
    update_config(cfg, args)

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )
    
    torchinfo.summary(
        model, 
        input_size=(1, 3, 256, 256), depth=3
    )
    
    if args.checkpoint is not None:
        if args.load_messy_checkpoint:
            load_messy_checkpoint(model, args.checkpoint)
        else:
            ckpt= torch.load(args.checkpoint)
            info = model.load_state_dict(ckpt, strict=True)
            print(info)
    

if __name__ == '__main__':
    main()
