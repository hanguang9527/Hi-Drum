import sys
from functools import reduce

import torch
from PIL import Image
from torch.nn.modules.module import _addindent


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, 'w')
        print(string, file=file)
        file.flush()

    return count


def save_pianoroll(path, onsets, frames, onset_threshold=0.5, frame_threshold=0.5, zoom=4):
    """
    Saves a piano roll plot

    Parameters
    ----------
    path: str
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    zoom: int
    """
    with torch.no_grad():  # Reduce memory usage
        # GPUOptimization: inGPUAll threshold and logic operations are performed on
        device = onsets.device
        
        # useboolType for more efficient binarization operations
        onsets_binary = (onsets.t() > onset_threshold)
        frames_binary = (frames.t() > frame_threshold)
        both = onsets_binary | frames_binary  # Use logical OR operation
        
        # Convert touint8and invert the colors (1Change0,0Change255)
        onsets_binary = (~onsets_binary).to(torch.uint8) * 255
        frames_binary = (~frames_binary).to(torch.uint8) * 255
        both = (~both).to(torch.uint8) * 255
        
        # GPUOptimization: inGPUStack image data on top and flip
        image_tensor = torch.stack([onsets_binary, frames_binary, both], dim=2).flip(0)
        
        # GPUOptimization: Usetorch.nn.functional.interpolateZoom
        if zoom != 1:
            # Rearrange dimensions to fitinterpolatefunction:[H, W, C] -> [1, C, H, W]
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).float()
            # existGPUbilinear interpolation scaling
            image_tensor = torch.nn.functional.interpolate(
                image_tensor, 
                size=(image_tensor.shape[2], image_tensor.shape[3] * zoom),
                mode='nearest'  # Preserve pixel sharpness using nearest neighbor interpolation
            )
            # Convert back to original format:[1, C, H, W] -> [H, W, C]
            image_tensor = image_tensor.squeeze(0).permute(1, 2, 0).to(torch.uint8)
        
        # Only converted at the end tonumpyandPILimage
        image_array = image_tensor.cpu().numpy()
        image = Image.fromarray(image_array, 'RGB')
        image.save(path)


def save_pianoroll_batch_gpu(paths, onsets_batch, frames_batch, onset_threshold=0.5, frame_threshold=0.5, zoom=4):
    """
    GPUOptimized batch piano roll image saving function
    
    Parameters
    ----------
    paths: list of str
        Save path list
    onsets_batch: torch.FloatTensor, shape = [batch, frames, bins]
        batchonsetdata
    frames_batch: torch.FloatTensor, shape = [batch, frames, bins]
        batchframedata
    onset_threshold: float
        onsetthreshold
    frame_threshold: float
        framethreshold
    zoom: int
        scaling factor
    """
    with torch.no_grad():
        batch_size = onsets_batch.shape[0]
        device = onsets_batch.device
        
        # Batch process all images
        for i in range(batch_size):
            onsets = onsets_batch[i]
            frames = frames_batch[i]
            path = paths[i]
            
            # useboolType for more efficient binarization operations
            onsets_binary = (onsets.t() > onset_threshold)
            frames_binary = (frames.t() > frame_threshold)
            both = onsets_binary | frames_binary
            
            # Convert touint8and invert the colors
            onsets_binary = (~onsets_binary).to(torch.uint8) * 255
            frames_binary = (~frames_binary).to(torch.uint8) * 255
            both = (~both).to(torch.uint8) * 255
            
            # existGPUStack image data on top and flip
            image_tensor = torch.stack([onsets_binary, frames_binary, both], dim=2).flip(0)
            
            # GPUOptimization: Usetorch.nn.functional.interpolateZoom
            if zoom != 1:
                # Rearrange dimensions:[H, W, C] -> [1, C, H, W]
                image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).float()
                # existGPUzoom on
                image_tensor = torch.nn.functional.interpolate(
                    image_tensor, 
                    size=(image_tensor.shape[2], image_tensor.shape[3] * zoom),
                    mode='nearest'
                )
                # Convert back to original format:[1, C, H, W] -> [H, W, C]
                image_tensor = image_tensor.squeeze(0).permute(1, 2, 0).to(torch.uint8)
            
            # Convert tonumpyandPILimage and save
            image_array = image_tensor.cpu().numpy()
            image = Image.fromarray(image_array, 'RGB')
            image.save(path)


def create_pianoroll_tensor_gpu(onsets, frames, onset_threshold=0.5, frame_threshold=0.5, zoom=4):
    """
    GPUOptimized piano roll image tensor creation function, returning the processed image tensor without saving it directly
    
    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    zoom: int
    
    Returns
    -------
    torch.Tensor: processed image tensor,shape = [H, W, 3], the data type isuint8
    """
    with torch.no_grad():
        # useboolType for more efficient binarization operations
        onsets_binary = (onsets.t() > onset_threshold)
        frames_binary = (frames.t() > frame_threshold)
        both = onsets_binary | frames_binary
        
        # Convert touint8and invert the colors
        onsets_binary = (~onsets_binary).to(torch.uint8) * 255
        frames_binary = (~frames_binary).to(torch.uint8) * 255
        both = (~both).to(torch.uint8) * 255
        
        # existGPUStack image data on top and flip
        image_tensor = torch.stack([onsets_binary, frames_binary, both], dim=2).flip(0)
        
        # GPUOptimization: Usetorch.nn.functional.interpolateZoom
        if zoom != 1:
            # Rearrange dimensions:[H, W, C] -> [1, C, H, W]
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).float()
            # existGPUzoom on
            image_tensor = torch.nn.functional.interpolate(
                image_tensor, 
                size=(image_tensor.shape[2], image_tensor.shape[3] * zoom),
                mode='nearest'
            )
            # Convert back to original format:[1, C, H, W] -> [H, W, C]
            image_tensor = image_tensor.squeeze(0).permute(1, 2, 0).to(torch.uint8)
        
        return image_tensor
