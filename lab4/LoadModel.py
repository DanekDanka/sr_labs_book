# The script downloads model's weights
# Requirement: wget running on a Linux system 


# Import of modules
import os
import subprocess
from collections import OrderedDict
import glob

import torch


def load_model(model, lines, save_path, reload=False):
    # Load model's weights (baseline model format)
    
    if not os.path.exists(save_path):
        os.mkdir(save_path, mode=0o777)

    for line in lines:
        url     = line.strip()
        outfile = url.split('/')[-1]

        out = 0

        # Download files
        if not os.path.exists(os.path.join(save_path, outfile)) or reload:
            out = subprocess.call('wget %s -O %s/%s'%(url, save_path, outfile), shell=True)
        
        if out != 0:
            raise ValueError('Download failed %s. If download fails repeatedly, use alternate URL on the VoxCeleb website.'%url)

        print('File %s is downloaded.'%outfile)
        
    checkpoint = torch.load(os.path.join(save_path, 'baseline_v2_ap.model'), map_location='cpu')
    
    model_weight = OrderedDict()

    for key in checkpoint.keys():
        
        if '__S__' in key:
            model_weight[key[6:]] = checkpoint[key]
    
    # Load with strict=False to handle architecture differences
    missing_keys, unexpected_keys = model.load_state_dict(model_weight, strict=False)
    
    if missing_keys:
        print(f'Warning: {len(missing_keys)} missing keys (using random initialization)')
    if unexpected_keys:
        print(f'Info: {len(unexpected_keys)} unexpected keys (ignored)')
    
    return model


def load_model_from_lab3(model, model_path, epoch=None):
    # Load model trained in lab3 (with or without augmentation)
    # model_path: path to directory with .pth files (e.g., '../data/lab3_models_aug')
    # epoch: specific epoch to load (None = load latest)
    
    if not os.path.exists(model_path):
        raise ValueError(f'Model path does not exist: {model_path}')
    
    # Find all .pth files
    pth_files = glob.glob(os.path.join(model_path, '*.pth'))
    
    if not pth_files:
        raise ValueError(f'No .pth files found in {model_path}')
    
    if epoch is not None:
        # Load specific epoch
        model_file = os.path.join(model_path, f'lab3_model_{str(epoch).zfill(4)}.pth')
        if not os.path.exists(model_file):
            raise ValueError(f'Model file for epoch {epoch} not found: {model_file}')
    else:
        # Load latest epoch (highest number)
        pth_files.sort()
        model_file = pth_files[-1]
        print(f'Loading latest model: {os.path.basename(model_file)}')
    
    checkpoint = torch.load(model_file, map_location='cpu')
    
    # Check if it's lab3 format (has 'model' key) or baseline format (has '__S__' keys at top level)
    if 'model' in checkpoint:
        # Lab3 format - model saved as MainModel, which contains __S__ keys
        model_state_dict_raw = checkpoint['model']
        # Extract ResNet weights from MainModel (remove '__S__.' prefix)
        model_state_dict = OrderedDict()
        for key in model_state_dict_raw.keys():
            if key.startswith('__S__.'):
                # Remove '__S__.' prefix to get ResNet layer names
                model_state_dict[key[6:]] = model_state_dict_raw[key]
            elif not key.startswith('__L__.'):  # Ignore loss function weights
                # If no prefix, assume it's already ResNet format
                model_state_dict[key] = model_state_dict_raw[key]
    else:
        # Baseline format - extract from __S__ keys at top level
        model_state_dict = OrderedDict()
        for key in checkpoint.keys():
            if '__S__' in key:
                model_state_dict[key[6:]] = checkpoint[key]
    
    # Get current model's state dict
    current_state_dict = model.state_dict()
    
    # Filter compatible weights
    compatible_weights = OrderedDict()
    missing_keys = []
    unexpected_keys = []
    
    for key in model_state_dict.keys():
        if key in current_state_dict:
            # Check if shapes match
            if model_state_dict[key].shape == current_state_dict[key].shape:
                compatible_weights[key] = model_state_dict[key]
            else:
                unexpected_keys.append(f'{key} (shape mismatch: {model_state_dict[key].shape} vs {current_state_dict[key].shape})')
        else:
            unexpected_keys.append(key)
    
    # Find missing keys
    for key in current_state_dict.keys():
        if key not in compatible_weights:
            missing_keys.append(key)
    
    # Load compatible weights
    model.load_state_dict(compatible_weights, strict=False)
    
    # Print summary
    print(f'Loaded model from: {os.path.basename(model_file)}')
    if 'num_epoch' in checkpoint:
        print(f'Trained for {checkpoint["num_epoch"]} epochs')
    
    if missing_keys:
        print(f'\nWarning: Missing keys (will use random initialization): {len(missing_keys)} keys')
        if len(missing_keys) <= 10:
            for key in missing_keys:
                print(f'  - {key}')
        else:
            for key in missing_keys[:10]:
                print(f'  - {key}')
            print(f'  ... and {len(missing_keys) - 10} more')
    
    if unexpected_keys:
        print(f'\nInfo: Ignored unexpected keys: {len(unexpected_keys)} keys')
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f'  - {key}')
        else:
            for key in unexpected_keys[:10]:
                print(f'  - {key}')
            print(f'  ... and {len(unexpected_keys) - 10} more')
    
    return model