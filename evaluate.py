"""
Model evaluation script

This script is used to evaluate the performance of a trained music transcription model. It uses multiple evaluation metrics to measure the model
Performance on tasks such as note detection, frame detection, and velocity prediction.

Main functions:
1. Load the trained model and test data set
2. Make predictions on test data
3. Calculate various evaluation indicators (precision rate, recall rate, F1 score, etc.)
4. Optionally save visual images and MIDI files of prediction results

Evaluation indicators include:
- Note level evaluation (with/without offset)
- Note level evaluation (including velocity information)
- Frame level evaluation
- Various loss function values
"""

import argparse
import collections
import json
import os
import sys
from collections import defaultdict

import numpy as np
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription import match_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
import mir_eval.onset
from mir_eval.util import midi_to_hz
from scipy.stats import hmean
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import re

import onsets_and_frames.dataset as dataset_module
from onsets_and_frames import *
from onsets_and_frames.transcriber import run_transcription_batch


def parse_model_params_from_path(path):
    """Parse model parameters from file path"""
    basename = os.path.basename(os.path.dirname(path)) # Get directory name
    
    # default value
    encoder_model = 'MERT_330m'
    decoder_model = 'Qwen3-0.6B'
    encoder_layer = 'all'
    decoder_layer = 'all'
    head_mode = 'independent'
    enable_branch1 = True
    enable_branch2 = True
    enable_branch3 = True

    # Why: custom logdirs may not encode architecture in the path; prefer training config.json for exact rebuild.
    try:
        config = load_saved_config_from_model_file(path)
        if config is not None:
            encoder_model = config.get('encoder_model', encoder_model)
            decoder_model = config.get('decoder_model', decoder_model)
            encoder_layer = config.get('encoder_layer_selection', encoder_layer)
            decoder_layer = config.get('decoder_layer_selection', decoder_layer)
            head_mode = config.get('head_mode', head_mode)
            enable_branch1 = config.get('enable_branch1', enable_branch1)
            enable_branch2 = config.get('enable_branch2', enable_branch2)
            enable_branch3 = config.get('enable_branch3', enable_branch3)
            decoder_backend = config.get('decoder_backend', 'llm')
            encoder_freeze = config.get('encoder_freeze', False)
            decoder_freeze_model = config.get('decoder_freeze_model', True)
            return (
                encoder_model,
                encoder_layer,
                decoder_model,
                decoder_layer,
                head_mode,
                enable_branch1,
                enable_branch2,
                enable_branch3,
                decoder_backend,
                encoder_freeze,
                decoder_freeze_model,
            )
    except Exception as e:
        print(f"Warning: Failed to parse config near {path}: {e}")
    
    try:
        # Regular matching: encoder_{...}_layer{...}_decoder_{...}_layer{...}_dataset
        # Note: The model name may contain underscores, and the layer may contain underscores.
        match = re.search(r'encoder_(.*?)_layer(.*?)_decoder_(.*?)_layer(.*?)_dataset', basename)
        if match:
            encoder_model = match.group(1)
            encoder_layer_str = match.group(2)
            decoder_model = match.group(3)
            decoder_layer_str = match.group(4)
            
            # Restore layer string (layerall -> all, layer0_1_2 -> 0,1,2)
            # Note: In train.py, the string corresponding to layerall is 'all'
            # encoder_layer_str = encoder_layer_selection.replace(',', '_') if encoder_layer_selection != 'all' else 'all'
            
            encoder_layer = encoder_layer_str if encoder_layer_str == 'all' else encoder_layer_str.replace('_', ',')
            decoder_layer = decoder_layer_str if decoder_layer_str == 'all' else decoder_layer_str.replace('_', ',')

        branch_match = re.search(r'_([TF]{3})_(HeadsPaper|HeadsInd)_', basename)
        if branch_match:
            branch_config = branch_match.group(1)
            enable_branch1, enable_branch2, enable_branch3 = [flag == 'T' for flag in branch_config]
            head_mode = 'paper_hierarchical' if branch_match.group(2) == 'HeadsPaper' else 'independent'
        else:
            if 'HeadsPaper' in basename:
                head_mode = 'paper_hierarchical'
            elif 'HeadsInd' in basename:
                head_mode = 'independent'
            
    except Exception as e:
        print(f"Warning: Failed to parse model params from path: {e}")
        
    return (
        encoder_model,
        encoder_layer,
        decoder_model,
        decoder_layer,
        head_mode,
        enable_branch1,
        enable_branch2,
        enable_branch3,
        'llm',
        False,
        True,
    )


def load_model_from_checkpoint(model_file, device):
    """Load the model, support loading objects directly or rebuilding from state_dict"""
    try:
        # Try loading the entire model directly
        model = torch.load(model_file, map_location=device, weights_only=False)
        # Check if it is state_dict
        if isinstance(model, dict) or isinstance(model,  collections.OrderedDict): 
            raise AttributeError("Loaded object is a state_dict")
        return model.eval()
    except (AttributeError, RuntimeError, TypeError) as e:
        print(f"Direct load failed ({e}), attempting to reconstruct model from params...")
        
        # Parse parameters
        (
            encoder_model,
            encoder_layer,
            decoder_model,
            decoder_layer,
            head_mode,
            enable_branch1,
            enable_branch2,
            enable_branch3,
            decoder_backend,
            encoder_freeze,
            decoder_freeze_model,
        ) = parse_model_params_from_path(model_file)
            
        print(
            f"Parsed params: Encoder={encoder_model}, E-Layer={encoder_layer}, "
            f"Decoder={decoder_model}, D-Layer={decoder_layer}, "
            f"Backend={decoder_backend}, encFZ={encoder_freeze}, decFZ={decoder_freeze_model}, "
            f"HeadMode={head_mode}, Adapter={''.join(['T' if b else 'F' for b in [enable_branch1, enable_branch2, enable_branch3]])}"
        )
        
        # Build configuration
        model_complexity = 48
        
        # Instantiate the model
        model = OnsetsAndFrames(
            output_features=MAX_MIDI - MIN_MIDI + 1,
            model_complexity=model_complexity,
            encoder_model=encoder_model,
            encoder_layer_selection=encoder_layer,
            # Why: evaluation must reuse the decoder layer-selection used at training time.
            decoder_layer_selection=decoder_layer,
            decoder_model=decoder_model,
            decoder_freeze_model=decoder_freeze_model,
            decoder_backend=decoder_backend,
            encoder_freeze=encoder_freeze,
            device=device,
            enable_branch1=enable_branch1,
            enable_branch2=enable_branch2,
            enable_branch3=enable_branch3,
            head_mode=head_mode,
        )
        
        # Load parameters
        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict)
        return model.to(device).eval()


def gpu_optimized_postprocess(p_ref, i_ref, v_ref, p_est, i_est, v_est, 
                             t_ref, f_ref, t_est, f_est, scaling, device):
    """
    GPU-optimized post-processing functions to reduce CPU-GPU data transfer
    
    Parameters
    ----------
    p_ref, i_ref, v_ref: reference note data
    p_est, i_est, v_est: predicted note data
    t_ref, f_ref, t_est, f_est: frame data
    scaling: time scaling factor
    device: GPU device
    
    Returns
    -------
    Processed data, ready for evaluation
    """
    with torch.no_grad():
        # Move data to GPU for batch processing
        if len(i_ref) > 0:
            i_ref_tensor = torch.from_numpy(i_ref).float().to(device)  # Convert data to device - supports GPU acceleration
            i_ref_scaled = (i_ref_tensor * scaling).cpu().numpy().reshape(-1, 2)
        else:
            i_ref_scaled = np.array([]).reshape(0, 2)
            
        if len(i_est) > 0:
            i_est_tensor = torch.from_numpy(i_est).float().to(device)  # Convert data to device - supports GPU acceleration
            i_est_scaled = (i_est_tensor * scaling).cpu().numpy().reshape(-1, 2)
        else:
            i_est_scaled = np.array([]).reshape(0, 2)
        
        # Batch MIDI to frequency conversion
        if len(p_ref) > 0:
            p_ref_tensor = torch.from_numpy(p_ref).float().to(device)  # Convert data to device - supports GPU acceleration
            # Use GPU for batch calculation: f = 440 * 2^((midi-69)/12)
            p_ref_freq = (440.0 * torch.pow(2.0, (MIN_MIDI + p_ref_tensor - 69) / 12.0)).cpu().numpy()
        else:
            p_ref_freq = np.array([])
            
        if len(p_est) > 0:
            p_est_tensor = torch.from_numpy(p_est).float().to(device)  # Convert data to device - supports GPU acceleration
            p_est_freq = (440.0 * torch.pow(2.0, (MIN_MIDI + p_est_tensor - 69) / 12.0)).cpu().numpy()
        else:
            p_est_freq = np.array([])
        
        # Batch processing frame time scaling
        t_ref_scaled = (torch.from_numpy(t_ref).double().to(device) * scaling).cpu().numpy()
        t_est_scaled = (torch.from_numpy(t_est).double().to(device) * scaling).cpu().numpy()
        
        # Batch processing frequency conversion
        f_ref_freq = []
        for freqs in f_ref:
            if len(freqs) > 0:
                freqs_tensor = torch.from_numpy(freqs).float().to(device)
                freq_hz = (440.0 * torch.pow(2.0, (MIN_MIDI + freqs_tensor - 69) / 12.0)).cpu().numpy()
                f_ref_freq.append(freq_hz)
            else:
                f_ref_freq.append(np.array([]))
        
        f_est_freq = []
        for freqs in f_est:
            if len(freqs) > 0:
                freqs_tensor = torch.from_numpy(freqs).float().to(device)
                freq_hz = (440.0 * torch.pow(2.0, (MIN_MIDI + freqs_tensor - 69) / 12.0)).cpu().numpy()
                f_est_freq.append(freq_hz)
            else:
                f_est_freq.append(np.array([]))
    
    return (i_ref_scaled, p_ref_freq, i_est_scaled, p_est_freq, 
            t_ref_scaled, f_ref_freq, t_est_scaled, f_est_freq)


def extract_onset_events(onsets, onset_threshold=0.5):
    """
    Extract the onset events (time and pitch) from the onset prediction for MIREX standard onset evaluation
    
    Parameters
    ----------
    onsets: torch.FloatTensor or np.ndarray, shape = [frames, bins]
        onset predicted value
    onset_threshold: float
        onset threshold
        
    Returns
    -------
    onset_times: np.ndarray
        onset time (frame index)
    onset_pitches: np.ndarray
        onset pitch (bin index)
    """
    # Convert to numpy and binarize
    if isinstance(onsets, torch.Tensor):
        onsets_binary = (onsets > onset_threshold).cpu().numpy()
    else:
        onsets_binary = (onsets > onset_threshold).astype(np.uint8)
    
    # Find all onset transition points (0->1)
    # Fill the timeline with zeros to detect jumps in the first frame
    onsets_padded = np.pad(onsets_binary, ((1, 0), (0, 0)), mode='constant')
    
    #Detect rising edge: current frame=1 and previous frame=0
    onset_diff = (onsets_padded[1:] == 1) & (onsets_padded[:-1] == 0)
    
    # Get all onset events (time, pitch)
    onset_times, onset_pitches = np.nonzero(onset_diff)
    
    return onset_times, onset_pitches


# Minimum value for numerical stability
eps = sys.float_info.epsilon

# Frozen evaluation protocol (ICASSP Phase A). Keep in sync with runs/_PROTOCOL.md.
PROTOCOL_ID = 'hidrum-eval-v1-2026-08-06'
PROTOCOL_ONSET_TOLERANCE_SEC = 0.05  # mir_eval onset / note onset window
PROTOCOL_FRAME_DECODE = 'symmetric_extract_notes_to_frames'

#Dataset configuration dictionary, mapping the data set name to the corresponding data set class
DATASET_CONFIG = {
    'Groove': dataset_module.Groove,
    'groove': dataset_module.Groove,  # Support lowercase
}


def load_saved_config_from_model_file(model_file):
    """Load Sacred config.json next to a checkpoint if it exists."""
    config_paths = [
        os.path.join(os.path.dirname(model_file), 'config.json'),
        os.path.join(os.path.dirname(model_file), '1', 'config.json'),
        os.path.join(os.path.dirname(model_file), '2', 'config.json'),
        os.path.join(os.path.dirname(model_file), '3', 'config.json'),
    ]
    for config_path in config_paths:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    return None


def resolve_dataset_path(model_file, dataset_path=None):
    """Prefer explicit --dataset-path; else Sacred config; else None (loader default)."""
    if dataset_path:
        return dataset_path
    saved_config = load_saved_config_from_model_file(model_file)
    if saved_config is not None and saved_config.get('dataset_path'):
        return saved_config['dataset_path']
    return None


def print_protocol_banner(onset_threshold, frame_threshold, dataset_path, dataset_group):
    """Print the frozen Phase-A evaluation protocol so every run is self-describing."""
    print('=' * 72)
    print(f'Hi-Drum eval protocol: {PROTOCOL_ID}')
    print(f'  sample_rate={SAMPLE_RATE}  hop_length={HOP_LENGTH}  '
          f'feature_hz={SAMPLE_RATE // HOP_LENGTH}')
    print(f'  midi_range=[{MIN_MIDI}, {MAX_MIDI}]  '
          f'onset_threshold={onset_threshold}  frame_threshold={frame_threshold}')
    print(f'  onset_tolerance_sec={PROTOCOL_ONSET_TOLERANCE_SEC}  '
          f'frame_decode={PROTOCOL_FRAME_DECODE}')
    print(f'  dataset_path={dataset_path}  group={dataset_group}')
    print('  required_metrics: note/f1, onset/f1, frame/f1 '
          '(velocity optional on labeled sets)')
    print('=' * 72)


def summarize_metrics(metrics):
    """Mean/std for metric/* keys; suitable for JSON export."""
    summary = {
        'protocol_id': PROTOCOL_ID,
        'frame_decode': PROTOCOL_FRAME_DECODE,
        'onset_tolerance_sec': PROTOCOL_ONSET_TOLERANCE_SEC,
        'metrics': {},
    }
    for key, values in metrics.items():
        if not key.startswith('metric/'):
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary['metrics'][key] = {
            'mean': float(np.mean(arr)) if arr.size else None,
            'std': float(np.std(arr)) if arr.size else None,
            'n': int(arr.size),
        }
    return summary


def write_results_json(path, summary, extra=None):
    payload = dict(summary)
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f'Wrote results JSON: {path}')


def evaluate(data, model, onset_threshold=0.5, frame_threshold=0.5, save_path=None):
    """
    Evaluate the performance of the model on the data set
    
    This function makes predictions for each sample in the dataset and then calculates various evaluation metrics.
    It compares model predictions to ground truth labels, calculating note-level and frame-level performance metrics.
    
    Args:
        data: Data set iterator, each element contains audio features and corresponding labels
        model: the model to evaluate
        onset_threshold (float): Threshold for note onset detection, default 0.5
        frame_threshold (float): Threshold for note frame detection, default 0.5
        save_path (str, optional): If specified, the visualization image and MIDI file of the prediction results will be saved
        
    Returns:
        dict: A dictionary containing various evaluation indicators, the key name format is 'metric/category/name'
    """
    # Use defaultdict to store all evaluation metrics
    metrics = defaultdict(list)
    
    # Get device information for GPU optimization
    device = next(model.parameters()).device

    # Loop through each sample in the dataset
    for label in data:
        # Use the model to make predictions
        pred, losses = run_transcription_batch(model, label)

        # Collect loss function values
        for key, loss in losses.items():
            metrics[key].append(loss.item())

        # GPU optimization: post-process the prediction results (activate, remove batch dimension)
        with torch.no_grad():
            for key, value in pred.items():
                value.squeeze_(0).relu_()

        # Extract note information (pitch, time interval, velocity) from real tags
        # Remove the batch dimension (squeeze dimension 0)
        p_ref, i_ref, v_ref = extract_notes(label['onset'].squeeze(0), label['frame'].squeeze(0), label['velocity'].squeeze(0))
        # Extract note information from prediction results
        p_est, i_est, v_est = extract_notes(pred['onset'], pred['frame'], pred['velocity'], 
                                          onset_threshold, frame_threshold)

        # Frame-level evaluation: ref and est both use the same note-decode pipeline so
        # metrics stay symmetric (GT direct piano-roll caused ~0.40 F1 with decode est).
        ref_frame_shape = label['frame'].squeeze(0).shape
        t_ref, f_ref = notes_to_frames(p_ref, i_ref, ref_frame_shape)
        t_est, f_est = notes_to_frames(p_est, i_est, pred['frame'].shape)

        # Time scaling factor: Convert frame index to actual time (seconds)
        scaling = HOP_LENGTH / SAMPLE_RATE

        # GPU optimization: use batch post-processing functions
        (i_ref_scaled, p_ref_freq, i_est_scaled, p_est_freq, 
         t_ref_scaled, f_ref_freq, t_est_scaled, f_est_freq) = gpu_optimized_postprocess(
            p_ref, i_ref, v_ref, p_est, i_est, v_est, 
            t_ref, f_ref, t_est, f_est, scaling, device
        )

        # -------------------------------------------------------
        # MIREX standard: Onset level evaluation (Event level ±50ms tolerance)
        # -------------------------------------------------------
        # Extract onset time (seconds)
        ref_onset_times, _ = extract_onset_events(label['onset'].squeeze(0), onset_threshold)
        est_onset_times, _ = extract_onset_events(pred['onset'], onset_threshold)
        
        ref_onsets_sec = ref_onset_times * scaling
        est_onsets_sec = est_onset_times * scaling
        
        # Use mir_eval.onset.f_measure (±50ms window)
        # Note: mir_eval.onset.f_measure returns (F1, Precision, Recall)
        if len(ref_onsets_sec) == 0 and len(est_onsets_sec) == 0:
            onset_f1, onset_precision, onset_recall = 1.0, 1.0, 1.0
        else:
            onset_f1, onset_precision, onset_recall = mir_eval.onset.f_measure(
                ref_onsets_sec, est_onsets_sec, window=PROTOCOL_ONSET_TOLERANCE_SEC
            )
        
        # Record the onset indicator
        metrics['metric/onset/precision'].append(onset_precision)
        metrics['metric/onset/recall'].append(onset_recall)
        metrics['metric/onset/f1'].append(onset_f1)
        
        # Note level evaluation (offset not considered)
        p, r, f, o = evaluate_notes(i_ref_scaled, p_ref_freq, i_est_scaled, p_est_freq, offset_ratio=None)
        metrics['metric/note/precision'].append(p)
        metrics['metric/note/recall'].append(r)
        metrics['metric/note/f1'].append(f)
        metrics['metric/note/overlap'].append(o)

        # Note level evaluation (taking into account offsets)
        p, r, f, o = evaluate_notes(i_ref_scaled, p_ref_freq, i_est_scaled, p_est_freq)
        metrics['metric/note-with-offsets/precision'].append(p)
        metrics['metric/note-with-offsets/recall'].append(r)
        metrics['metric/note-with-offsets/f1'].append(f)
        metrics['metric/note-with-offsets/overlap'].append(o)

        # Note level evaluation (including velocity, not considering offset)
        p, r, f, o = evaluate_notes_with_velocity(i_ref_scaled, p_ref_freq, v_ref, i_est_scaled, p_est_freq, v_est,
                                                  offset_ratio=None, velocity_tolerance=0.1)
        metrics['metric/note-with-velocity/precision'].append(p)
        metrics['metric/note-with-velocity/recall'].append(r)
        metrics['metric/note-with-velocity/f1'].append(f)
        metrics['metric/note-with-velocity/overlap'].append(o)

        # Note level evaluation (including velocity and offset)
        p, r, f, o = evaluate_notes_with_velocity(i_ref_scaled, p_ref_freq, v_ref, i_est_scaled, p_est_freq, v_est, 
                                                  velocity_tolerance=0.1)
        metrics['metric/note-with-offsets-and-velocity/precision'].append(p)
        metrics['metric/note-with-offsets-and-velocity/recall'].append(r)
        metrics['metric/note-with-offsets-and-velocity/f1'].append(f)
        metrics['metric/note-with-offsets-and-velocity/overlap'].append(o)

        # -------------------------------------------------------
        # Custom extension metrics: individually evaluate the accuracy of Velocity (MSE, MAE)
        # (Non-MIREX standard, but used for detailed analysis of strength prediction performance)
        # Calculate the Velocity error only for successfully matched notes
        # -------------------------------------------------------
        # Use mir_eval to find matching notes (based on Onset and Pitch)
        # Note: Here we use the same matching criteria as evaluate_notes (offset_ratio=None)
        matching = match_notes(i_ref_scaled, p_ref_freq, i_est_scaled, p_est_freq, offset_ratio=None)
        
        # -------------------------------------------------------
        # Custom extended metrics: calculate individual F1 scores for each MIDI note
        # (Non-MIREX standard, but used to analyze transcription performance for different drum instruments)
        # -------------------------------------------------------
        # Get the matching index set
        matched_ref_indices = set(m[0] for m in matching)
        matched_est_indices = set(m[1] for m in matching)
        
        # Get all notes appearing in the current sample (including real labels and predicted results)
        unique_notes = set(p_ref) | set(p_est)
        
        for note_idx in unique_notes:
            # Revert to MIDI note number
            note_num = int(note_idx) + MIN_MIDI
            
            # Calculate the TP (True Positive) of this note
            # The number of reference notes that belong to this note and are in the matching set
            ref_indices_k = [i for i, p in enumerate(p_ref) if p == note_idx]
            tp = sum(1 for i in ref_indices_k if i in matched_ref_indices)
            
            # Calculate FN (False Negative)
            # Number of reference notes that belong to this note but are not matched
            fn = len(ref_indices_k) - tp
            
            # Calculate FP (False Positive)
            # The number of predicted notes that were predicted to be this note but were not matched
            est_indices_k = [i for i, p in enumerate(p_est) if p == note_idx]
            fp = sum(1 for i in est_indices_k if i not in matched_est_indices)
            
            # Calculate F1 score
            if tp + fp + fn > 0:
                f1 = 2 * tp / (2 * tp + fp + fn)
            else:
                f1 = 0.0
            
            # Record to metrics
            # Note: TensorBoard will automatically record these newly added metrics
            metrics[f'metric/note/f1_{note_num}'].append(f1)

        if len(matching) > 0:
            # Decompress matching index
            ref_indices, est_indices = zip(*matching)
            
            # Extract the Velocity of matching notes
            # v_ref and v_est should be numpy arrays
            v_ref_matched = v_ref[list(ref_indices)]
            v_est_matched = v_est[list(est_indices)]
            
            # Calculate error
            diff = v_est_matched - v_ref_matched
            abs_diff = np.abs(diff)
            
            # Record indicators
            metrics['metric/velocity/mae'].append(np.mean(abs_diff))
            metrics['metric/velocity/mse'].append(np.mean(diff ** 2))
            
            # Calculate correlation coefficient (at least 2 points required)
            if len(v_ref_matched) > 1:
                # Avoid division-by-zero errors caused by zero variance
                if np.std(v_ref_matched) > 1e-6 and np.std(v_est_matched) > 1e-6:
                    corr = np.corrcoef(v_ref_matched, v_est_matched)[0, 1]
                    metrics['metric/velocity/corr'].append(corr)

        # Frame level evaluation
        frame_metrics = evaluate_frames(t_ref_scaled, f_ref_freq, t_est_scaled, f_est_freq)
        # Calculate frame-level F1 score (harmonic mean)
        metrics['metric/frame/f1'].append(hmean([frame_metrics['Precision'] + eps, 
                                               frame_metrics['Recall'] + eps]) - eps)

        # Collect other frame level metrics
        for key, loss in frame_metrics.items():
            metrics['metric/frame/' + key.lower().replace(' ', '_')].append(loss)

        # If the save path is specified, save the prediction results
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            # Processing path: if it is a list, take the first element, if it is a byte string, decode it
            path = label['path']
            if isinstance(path, (list, tuple)):
                path = path[0]
            if isinstance(path, bytes):
                path = path.decode('utf-8')
            
            # Save the piano roll image of the real label
            label_path = os.path.join(save_path, os.path.basename(path) + '.label.png')
            save_pianoroll(label_path, label['onset'].squeeze(0), label['frame'].squeeze(0))
            # Save the piano roll graph of the prediction results
            pred_path = os.path.join(save_path, os.path.basename(path) + '.pred.png')
            save_pianoroll(pred_path, pred['onset'], pred['frame'])
            # Save the MIDI file of the prediction results (using drum sounds)
            midi_path = os.path.join(save_path, os.path.basename(path) + '.pred.mid')
            # Data format conversion (consistent with transcribe.py)
            # Time scaling: Convert frame index to actual time
            i_est_scaled = (i_est * scaling).reshape(-1, 2)
            # Convert MIDI note symbols to frequency (Hz)
            p_est_hz = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])
            save_midi(midi_path, p_est_hz, i_est_scaled, v_est, is_drum=True)

    return metrics


def evaluate_file(model_file, dataset, dataset_group, sequence_length, save_path,
                  onset_threshold, frame_threshold, device, dataset_path=None):
    dataset_class = DATASET_CONFIG[dataset]
    kwargs = {'sequence_length': sequence_length}
    resolved_path = resolve_dataset_path(model_file, dataset_path)
    if resolved_path:
        kwargs['path'] = resolved_path
    if dataset_group is not None:
        kwargs['groups'] = [dataset_group]
    dataset = dataset_class(**kwargs)

    model = load_model_from_checkpoint(model_file, device)
    summary(model)

    metrics = evaluate(tqdm(dataset), model, onset_threshold, frame_threshold, save_path)

    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')


def main(model_file, dataset, dataset_group, save_path=None, batch_size=8,
         sequence_length=120000, onset_threshold=0.5, frame_threshold=0.5, device=None,
         dataset_path=None, results_json=None):
    """
    Evaluate the main function and support external device parameters to be passed in
    
    Args:
        model_file: trained model file path
        dataset: dataset name (such as Groove, etc.)
        dataset_group: Data set grouping (such as test, validation, etc.)
        save_path: path to save prediction results (optional)
        batch_size: evaluation batch size. The current metric extraction logic assumes batch_size=1.
        sequence_length: sequence length, default is 120000 (consistent with training)
        onset_threshold: note onset detection threshold, default is 0.5
        frame_threshold: note frame detection threshold, default is 0.5
        device: computing device, if None, use automatic detection
        dataset_path: optional absolute dataset root (overrides Sacred config dataset_path)
        results_json: optional path to write mean/std metric summary JSON
    """
    # Keep the CLI dataset name; the loader object below reuses the same variable name historically.
    dataset_name = dataset

    # If device is not specified, use automatic detection
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Computing device - supports GPU acceleration, uses auto-detection (compatibility mode)

    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print(f"Use equipment: {device}")
    resolved_path = resolve_dataset_path(model_file, dataset_path)
    print_protocol_banner(onset_threshold, frame_threshold, resolved_path, dataset_group)
    
    # Perform evaluation in gradient-free mode, saving memory and increasing inference speed
    with torch.no_grad():
        try:
            # Load model (using enhanced loading function)
            model = load_model_from_checkpoint(model_file, device)
            
            # Get the encoder_model configuration from the model
            encoder_model = getattr(model, 'encoder_model', '95M')  #The default value is 95M
            print(f"Frontend model detected from model: {encoder_model}")
            
            # Initialize the data set and pass the correct encoder_model
            # Note: Some data set classes may not accept device and encoder_model parameters, adjust according to the situation
            try:
                dataset_kwargs = {
                    'groups': [dataset_group],
                    'sequence_length': sequence_length,
                }
                if resolved_path:
                    dataset_kwargs['path'] = resolved_path
                dataset = DATASET_CONFIG[dataset_name](**dataset_kwargs)
            except TypeError:
                print("Dataset initialization fallback triggered, retrying with minimal args...")
                dataset = DATASET_CONFIG[dataset_name](groups=[dataset_group], sequence_length=sequence_length)
            
            if batch_size != 1:
                print(
                    f"Warning: current evaluation/postprocess code assumes batch_size=1, "
                    f"forcing batch_size from {batch_size} to 1"
                )
                batch_size = 1

            # Why: metric extraction currently squeezes dim 0; force batch_size=1 until true multi-sample postprocess exists.
            # Why: eval is batch_size=1 and often CPU/decode bound; prefetch with workers
            # to keep the GPU fed when packing multiple eval processes per card.
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=(str(device).startswith('cuda')),
                persistent_workers=True,
            )

            # Perform evaluation
            metrics = evaluate(loader, model, onset_threshold, frame_threshold, save_path)

            # Calculate and output the average indicator
            for key, values in metrics.items():
                if key.startswith('metric/'):
                    # Calculate the average and output
                    _, category, name = key.split('/')
                    print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')
                else:
                    # Output loss function value
                    print(f'{key:>60}: {np.mean(values):.3f} ± {np.std(values):.3f}')

            if results_json:
                summary = summarize_metrics(metrics)
                write_results_json(
                    results_json,
                    summary,
                    extra={
                        'model_file': os.path.abspath(model_file),
                        'dataset_name': dataset_name,
                        'dataset_group': dataset_group,
                        'dataset_path': resolved_path,
                        'sequence_length': sequence_length,
                        'onset_threshold': onset_threshold,
                        'frame_threshold': frame_threshold,
                        'device': str(device),
                        'n_samples': len(dataset),
                    },
                )
        except Exception as e:
            print(f"An error occurred during evaluation: {e}")
            raise


if __name__ == '__main__':
    """
    Main program entrance
    
    This script can be run from the command line and used to evaluate a trained music transcription model.
    It loads the specified model and dataset, evaluates it and outputs various performance metrics.
    """
    #Create a command line parameter parser
    parser = argparse.ArgumentParser(description='Evaluating the performance of music transcription models')
    parser.add_argument('model_file', type=str, help='Trained model file path')
    parser.add_argument('dataset', type=str, help='Dataset name (such as Groove, etc.)')
    parser.add_argument('dataset_group', type=str, help='Dataset grouping (such as test, validation, etc.)')
    parser.add_argument('--save-path', type=str, default=None, 
                       help='Path to save prediction results (optional)')
    parser.add_argument('--batch-size', type=int, default=8, 
                       help='Batch size, default is 8')
    parser.add_argument('--sequence-length', type=int, default=120000, 
                       help='Sequence length, default is 120000 (consistent with training)')
    parser.add_argument('--onset-threshold', type=float, default=0.5, 
                       help='Note onset detection threshold, default is 0.5')
    parser.add_argument('--frame-threshold', type=float, default=0.5, 
                       help='Note frame detection threshold, default is 0.5')
    parser.add_argument('--device', type=str, default=None, 
                       help='Computing device (cuda or cpu), automatically selected if not specified')
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Dataset root override (e.g. groove_... or egmd_...). Required to score E-GMD when ckpt was trained with grouped data.',
    )
    parser.add_argument(
        '--results-json',
        type=str,
        default=None,
        help='Optional path to write protocol-tagged mean/std metrics JSON',
    )

    # Parse command line parameters
    args = parser.parse_args()

    # Call main function
    main(args.model_file, args.dataset, args.dataset_group, args.save_path,
         args.batch_size, args.sequence_length, args.onset_threshold, 
         args.frame_threshold, args.device, args.dataset_path, args.results_json)
