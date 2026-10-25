"""
Music Transcription Script - Simplified Version

This script is used to perform music transcription of audio files using the trained OnsetsAndFrames model.
Only three parameters need to be provided: training record folder, audio file path and result saving path.

The script will automatically read all model parameter configurations from the 1/config.json file in the training record folder.
There is no need to manually set complex parameters such as encoders, decoders, and branches.

How to use:
    python transcribe.py --training_record_dir <training record folder> --audio_path <audio file> --save_path <result save path>

Parameter description:
    --training_record_dir: training record folder path, including model-best-note.pt and 1/config.json
    --audio_path: Enter the audio file path (supports file, folder or wildcard mode, multiple can be specified)
    --save_path: path to the directory where the results are saved
    --device: optional, computing device (such as cuda:0, cuda:1, cpu)

Example:
    python transscribe.py --training_record_dir runs compared with ofa groove7class/encoder_MERT_330m_layerall_decoder_Qwen3-0.6B_layerall_dataset_groove_processed_10s_24kh7class_hop75hz_TTT_PostProc1_gpu1-251222-065354 --audio_path data/demo_dataset --save_path transcribe_results
"""

# 🎯 ==================== Parameter configuration area ====================
# Only need to modify these three parameters, other parameters will be automatically read from the training record folder

# 🎯 Training record folder path - the folder containing model-best-note.pt and 1/config.json
TRAINING_RECORD_DIR = 'Comparison of runs and ofa groove7class/encoder_MERT_330m_layerall_decoder_Qwen3-0.6B_layerall_dataset_groove_processed_10s_24kh7class_hop75hz_TTT_PostProc1_gpu1-251222-065354'

# 🎯 Audio file path
AUDIO_PATHS = ['data/groove_processed_10s_24kh7class/validation_data']  # Support file, folder or wildcard pattern

# 🎯 Result saving path
SAVE_PATH = 'transcribe_results_groove7class'

# 🎯 Audio segment duration (seconds)
# If None, use the default value in the training configuration (usually 10 seconds)
# If you specify a specific value (such as 5.0), this duration is forced to be used
SEGMENT_DURATION = None

# 🎯 ==================== End parameter configuration ====================

import os

import sys
import glob
import json
import importlib.util
import types
import argparse

import numpy as np
import soundfile
from mir_eval.util import midi_to_hz

from onsets_and_frames import *
from onsets_and_frames.constants import resolve_project_path


def load_saved_config(model_file):
    """
    Load the configuration saved during training from the model file path
    
    Args:
        model_file: model file path, for example 'runs/transcribe-MERT_95m-mlp-gpu0-250926-033647/model-best.pt'
    
    Returns:
        dict: Configuration parameters during training
        
    Raises:
        FileNotFoundError: If the config.json file cannot be found
        Exception: if an error occurs while loading the configuration file
    """
    # Infer the runs directory from the model file path
    model_dir = os.path.dirname(model_file)
    
    # Find the config.json file (maybe in a subdirectory)
    config_paths = [
        os.path.join(model_dir, 'config.json'),
        os.path.join(model_dir, '1', 'config.json'),
        os.path.join(model_dir, '2', 'config.json'),
        os.path.join(model_dir, '3', 'config.json'),
    ]
    
    config_file = None
    for path in config_paths:
        if os.path.exists(path):
            config_file = path
            break
    
    if config_file is None:
        raise FileNotFoundError(f"Error: in {model_dir} Not found inconfig.jsondocument. Please ensure that the model file path is correct and the configuration file is saved during training.")
    
    try:
        #Load configuration file
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"Successfully loaded training configuration: {config_file}")
        return config
        
    except Exception as e:
        raise Exception(f"Error loading configuration file {config_file}: {e}")


def load_source_snapshot(model_file, module_name):
    """
    Load the source code snapshot saved during training from the _sources folder of the model directory
    
    Args:
        model_file: model file path
        module_name: module name (such as 'constants', 'dataset', 'transcriber', etc.)
        
    Returns:
        module: loaded module object, returns None if failed
    """
    model_dir = os.path.dirname(model_file)
    sources_dir = os.path.join(model_dir, '_sources')
    
    if not os.path.exists(sources_dir):
        print(f"Source code snapshot directory not found: {sources_dir}")
        return None
    
    # Find the corresponding source code file
    for filename in os.listdir(sources_dir):
        if filename.startswith(f"{module_name}_") and filename.endswith('.py'):
            source_file = os.path.join(sources_dir, filename)
            try:
                # Dynamically load modules
                spec = importlib.util.spec_from_file_location(f"snapshot_{module_name}", source_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"Successfully loaded source code snapshot: {filename}")
                return module
            except Exception as e:
                print(f"Failed to load source code snapshot {filename}: {e}")
                return None
    
    print(f"Module not found {module_name} Source code snapshot of")
    return None


def load_and_process_audio(audio_path, sequence_length, device):
    """Load and preprocess FLAC/WAV audio, resampling to SAMPLE_RATE when needed."""
    audio, sr = soundfile.read(audio_path, dtype='int16')
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(
            audio.astype(np.float32),
            orig_sr=sr,
            target_sr=SAMPLE_RATE,
        )
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
        sr = SAMPLE_RATE

    duration = len(audio) / sr
    audio = torch.ShortTensor(audio)
    return audio, sr, duration


def transcribe(model, audio, saved_config=None, use_saved_config=False):
    """
    Use models to transcribe audio
    
    This function receives preprocessed audio through the trained OnsetsAndFrames model
    Carry out music transcription and predict the start, end, duration frame and velocity information of notes.
    
    For long audio, it is automatically chunked to avoid memory issues and dimensionality mismatches.
    
    Args:
        model: trained music transcription model (OnsetsAndFrames)
        audio (torch.Tensor): preprocessed audio tensor
        saved_config (dict, optional): Configuration saved during training, used to determine model parameters
        use_saved_config (bool): whether to use saved configuration and source code snapshots
        
    Returns:
        dict: A dictionary containing prediction results, with keys including:
            - 'onset': note onset prediction
            - 'frame': note frame prediction
            - 'velocity': note velocity prediction
    """
    # Use the unified OnsetsAndFrames model for inference
    with torch.no_grad():
        # Make sure the audio is on the correct device
        device = next(model.parameters()).device
        audio = audio.to(device)

        # Why: 训练侧是按“当前片段长度”做归一化；推理若先按整首音频长度归一化再切段，
        # 同一局部片段在长/短音频中的幅值会不同，造成训推输入分布失配。
        audio = audio.float().div_(len(audio) / 10)
        
        #Add batch dimension: [sequence_length] -> [1, sequence_length]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Directly use the forward method of the OnsetsAndFrames model
        # Data flow: audio → feature_extractor → qwen_decoder → MLP branches → predictions
        onset_pred, frame_pred, velocity_pred = model(audio)
        
        # Apply the sigmoid activation function to the logits and convert them into probability values ​​in the range of 0-1
        onset_pred = torch.sigmoid(onset_pred)
        frame_pred = torch.sigmoid(frame_pred)
        velocity_pred = torch.sigmoid(velocity_pred)

    # Organize the prediction results and remove the batch dimension
    predictions = {
            'onset': onset_pred.reshape((onset_pred.shape[1], onset_pred.shape[2])),
            'frame': frame_pred.reshape((frame_pred.shape[1], frame_pred.shape[2])),
            'velocity': velocity_pred.reshape((velocity_pred.shape[1], velocity_pred.shape[2]))
        }

    return predictions


def expand_audio_paths(audio_paths):
    """Expand FLAC/WAV paths, supporting folders, wildcards, and .txt file lists."""
    expanded_paths = []
    supported_ext = ('.flac', '.wav')

    for path in audio_paths:
        if os.path.isfile(path) and path.lower().endswith(supported_ext):
            expanded_paths.append(path)
        elif os.path.isfile(path) and path.endswith('.txt'):
            with open(path, 'r', encoding='utf-8') as list_file:
                for line in list_file:
                    item = line.strip()
                    if item and os.path.isfile(item):
                        expanded_paths.append(item)
        elif os.path.isdir(path):
            for pattern in ('**/*.flac', '**/*.wav', '**/*.FLAC', '**/*.WAV'):
                expanded_paths.extend(glob.glob(os.path.join(path, pattern), recursive=True))
        else:
            matched_files = glob.glob(path)
            expanded_paths.extend(
                sorted(f for f in matched_files if f.lower().endswith(supported_ext))
            )

    return sorted(set(expanded_paths))


def expand_flac_paths(flac_paths):
    return expand_audio_paths(flac_paths)


def transcribe_file(model_file, flac_paths, save_path, sequence_length,
                  onset_threshold, frame_threshold, segment_duration, device, use_saved_config=False, tempo=120,
                  audio_root=None):
    """
    Transcribe multiple audio files
    
    This is the main transcription function, which loads the trained model and then
    Carry out transcriptions and save the results as MIDI files and visualization images.
    
    Args:
        model_file (str): trained model file path
        flac_paths (list): List of FLAC audio file paths to be transcribed (supports files, folders or wildcards)
        save_path (str): result saving path
        sequence_length (int, optional): Audio sequence length, used for interception
        onset_threshold (float): Note onset detection threshold
        frame_threshold (float): note frame detection threshold
        device (str): computing device ('cpu' or 'cuda')
        use_saved_config (bool): Whether to use the configuration and source code snapshot saved during training
        tempo (int): tempo of the MIDI file (BPM)
    """
    #Expand input path, support folders and wildcards
    expanded_flac_paths = expand_flac_paths(flac_paths)
    
    if not expanded_flac_paths:
        print("Error: No FLAC/WAV files found", file=sys.stderr)
        return

    # Load the trained model and set it to evaluation mode
    # Use weights_only=False to support loading model files containing custom classes
    try:
        model = torch.load(model_file, map_location=device, weights_only=False).eval()
    except (AttributeError, RuntimeError, TypeError) as e:
        print(f"Direct load failed ({e}), attempting to reconstruct model from params...", file=sys.stderr)
        # If direct loading fails, try to rebuild the model (similar to the logic in evaluate.py)
        from evaluate import parse_model_params_from_path
        
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
        ) = parse_model_params_from_path(model_file)
            
        print(
            f"Parsed params: Encoder={encoder_model}, E-Layer={encoder_layer}, "
            f"Decoder={decoder_model}, D-Layer={decoder_layer}, HeadMode={head_mode}",
            file=sys.stderr
        )
        
        # Instantiate model - using latest schema parameters
        from onsets_and_frames import OnsetsAndFrames, MAX_MIDI, MIN_MIDI

        model = OnsetsAndFrames(
            output_features=MAX_MIDI - MIN_MIDI + 1,
            model_complexity=48,
            encoder_model=encoder_model,
            encoder_layer_selection=encoder_layer,
            decoder_layer_selection=decoder_layer,
            decoder_model=decoder_model,
            decoder_freeze_model=True,
            device=device,
            enable_branch1=enable_branch1,  # Parallel convolution branch
            enable_branch2=enable_branch2,  # dilated convolution branch
            enable_branch3=enable_branch3,  # Transformer branch
            head_mode=head_mode,  # multi-task heads mode
        )
        
        # Load parameters
        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device).eval()

    # Add a file path attribute to the model for subsequent loading of source code snapshots
    model.model_file = model_file
    
    summary(model)  #Print model structure summary
    
    # Load training configuration (if necessary)
    saved_config = None
    if use_saved_config:
        try:
            saved_config = load_saved_config(model_file)
        except (FileNotFoundError, Exception) as e:
            print(f"mistake:{e}", file=sys.stderr)
            sys.exit(1)  # Exit directly without using the default configuration

    # Traverse all audio files for transcription
    for i, flac_path in enumerate(expanded_flac_paths, 1):
        # 🎯 Extract BPM for each file individually (consistent with validation code)
        file_tempo = tempo  # Use global tempo as default value
        if file_tempo is None:
            try:
                filename_parts = os.path.basename(flac_path).split('_')
                if len(filename_parts) >= 3:
                    file_tempo = int(filename_parts[2])
                else:
                    file_tempo = 120
            except ValueError:
                file_tempo = 120
        
        #Load and preprocess audio
        audio, sr, duration = load_and_process_audio(flac_path, sequence_length, device)

        # Calculate parameters related to sampling points
        samples_per_segment = int(segment_duration * sr)
        total_samples = len(audio)
        min_samples = 10 * HOP_LENGTH  # Minimum sample number threshold (about 3200 samples)
        
        # Initial calculation number of segments (rounded up)
        num_segments = int(np.ceil(total_samples / samples_per_segment))
        
        # Lookahead merge strategy: check if the last paragraph is too short
        # If the last segment is too short and the total number of segments > 1, merge it into the previous segment
        remainder = total_samples % samples_per_segment
        if 0 < remainder < min_samples and num_segments > 1:
            print(f"Apply lookahead merging strategy: last segment length {remainder} Sample is too short (<{min_samples}), will be merged into the penultimate paragraph")
            num_segments -= 1  # Reduce the number of segments, the last segment will contain all the remaining audio

        # Initialize the dictionary to store the prediction results of all segments
        all_predictions = {'onset': [], 'frame': [], 'velocity': []}
        
        # Process audio segment by segment
        for i in range(num_segments):
            # Calculate the starting position of the current segment
            start_sample = i * samples_per_segment
            
            #Intercept audio segment
            if i == num_segments - 1:
                # If it is the last paragraph, go directly to the end (including the "tail" that may be merged)
                segment = audio[start_sample:]
            else:
                # Ordinary segmentation
                end_sample = (i + 1) * samples_per_segment
                segment = audio[start_sample:end_sample]
            
            # Double insurance: check if the current segment is still too short (for example, the total audio itself is extremely short)
            if len(segment) < min_samples:
                print(f"Warning: audio segment length {len(segment)} Sample is too short (needs at least {min_samples} sample), discard the segment")
                continue
            
            # Make transcription predictions for the current audio segment
            predictions = transcribe(model, segment, saved_config, use_saved_config)
            
            #Add the prediction results of the current segment to the list
            all_predictions['onset'].append(predictions['onset'])
            all_predictions['frame'].append(predictions['frame'])
            all_predictions['velocity'].append(predictions['velocity'])

        # Splice the prediction results of all segments into a complete tensor (time dimension splicing)
        if all_predictions['onset']:  # Ensure there are prediction results
            all_predictions['onset'] = torch.cat(all_predictions['onset'], dim=0)
            all_predictions['frame'] = torch.cat(all_predictions['frame'], dim=0)
            all_predictions['velocity'] = torch.cat(all_predictions['velocity'], dim=0)
            print(f"After splicing - velocity scope: [{all_predictions['velocity'].min():.4f}, {all_predictions['velocity'].max():.4f}]")
        else:
            # If there is no valid prediction result, create an empty tensor
            all_predictions['onset'] = torch.empty(0, 88)
            all_predictions['frame'] = torch.empty(0, 88)
            all_predictions['velocity'] = torch.empty(0, 88)
            print(f"After splicing - velocity is empty")

        # Extract notes from the prediction results: onset/frame threshold filtering to obtain note pitch, time interval, and intensity
        p_est, i_est, v_est = extract_notes(all_predictions['onset'], all_predictions['frame'], all_predictions['velocity'],
                                        onset_threshold, frame_threshold)
        
        # Calculate the time scaling factor: convert the time step of the model output into actual seconds
        # HOP_LENGTH (number of frame shift samples) / SAMPLE_RATE (sampling rate) = number of seconds corresponding to each time step
        scaling = HOP_LENGTH / SAMPLE_RATE

        # Convert the note time interval from time steps to seconds (reshape is N×2, each line is [start time, end time])
        i_est = (i_est * scaling).reshape(-1, 2)
        # Convert MIDI note numbers to actual frequencies (Hz): MIN_MIDI + midi is the absolute MIDI number
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

        #Create the save directory (preserve source folder layout when audio_root is set)
        if audio_root:
            rel_dir = os.path.dirname(os.path.relpath(flac_path, audio_root))
            output_dir = os.path.join(save_path, rel_dir) if rel_dir else save_path
        else:
            output_dir = save_path
        os.makedirs(output_dir, exist_ok=True)
        
        # Define the piano roll graph saving path (visualized prediction results)
        pred_path = os.path.join(output_dir, os.path.basename(flac_path) + '.pred.png')
        # Save the piano roll graph: visualize the onset and frame prediction results
        save_pianoroll(pred_path, all_predictions['onset'], all_predictions['frame'])
        
        #Define MIDI file saving path
        midi_path = os.path.join(output_dir, os.path.basename(flac_path) + '.pred.mid')
        
        # Save MIDI file: contains pitch, time, velocity information, specified speed (BPM)
        # Try to pass tempo if save_midi supports it, else fall back
        try:
            save_midi(midi_path, p_est, i_est, v_est, tempo=file_tempo)
        except TypeError:
            save_midi(midi_path, p_est, i_est, v_est)


def main(training_record_dir, audio_paths, save_path, device=None, segment_duration_arg=None, audio_root=None):
    """
    Transcribe the main function and support passing in external parameters
    
    Args:
        training_record_dir: training record folder path
        audio_paths: list of audio file paths
        save_path: result saving path
        device: computing device, if None, use automatic detection
        segment_duration_arg: audio segment duration (seconds), if it is None, use the value in the configuration file
    """
    # Check if the file exists
    model_file = os.path.join(training_record_dir, 'model-best-note.pt')
    config_file = os.path.join(training_record_dir, '1', 'config.json')
    
    # Check if the file exists
    if not os.path.exists(model_file):
        print(f"Error: model file does not exist: {model_file}")
        sys.exit(1)
    
    if not os.path.exists(config_file):
        print(f"Error: Configuration file does not exist: {config_file}")
        sys.exit(1)
    
    # Read parameters from configuration file
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"Successfully loaded training configuration: {config_file}")
    except Exception as e:
        print(f"Failed to load configuration file: {e}")
        sys.exit(1)
    
    # Read parameters from the configuration file (exclude device)
    encoder_model = config.get('encoder_model', 'MERT_330m')
    encoder_layer_selection = config.get('encoder_layer_selection', 'all')
    decoder_model = config.get('decoder_model', 'Qwen3-0.6B')
    decoder_freeze_model = config.get('decoder_freeze_model', True)
    use_unified_decoder = config.get('use_unified_decoder', True)
    enable_branch1 = config.get('enable_branch1', True)
    enable_branch2 = config.get('enable_branch2', True)
    enable_branch3 = config.get('enable_branch3', True)
    head_mode = config.get('head_mode', 'independent')
    sequence_length = config.get('sequence_length', None)
    onset_threshold = config.get('onset_threshold', 0.5)
    frame_threshold = config.get('frame_threshold', 0.5)
    
    # Prioritize using the segment_duration specified in the global configuration at the beginning of the script
    if segment_duration_arg is not None:
        segment_duration = segment_duration_arg
        print(f"Use a script to globally configure the specified audio segment duration: {segment_duration}Second")
    else:
        segment_duration = config.get('segment_duration', 10.0)  # Audio segment duration, default 10 seconds
        print(f"Use configuration files/Default audio segment duration: {segment_duration}Second")
        
    tempo = None  # Automatically extract from audio file names (will be extracted individually for each file as it is processed)
    
    print(f"useBPM: {tempo} (NoneIndicates that it will be automatically extracted from each file name)")
    
    # If device is not specified, use automatic detection
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Computing device - supports GPU acceleration, uses auto-detection (compatibility mode)
    
    use_saved_config = True  # Use the complete configuration saved during training
    
    # The project only uses Qwen front-end mode: MERT->Qwen cascade feature extraction pipeline
    
    # Qwen front-end configuration (only takes effect when saving configuration is not used)
    qwen_config = {
        "model_name": decoder_model,  # Qwen model name (supports multiple decoders)
        "local_path": resolve_project_path(f"model/{decoder_model}"),  # Local model path
        "layer_selection": "all",  # Qwen layer selection, consistent with training
        "freeze_model": decoder_freeze_model,  # Whether to freeze Qwen model parameters
        "use_unified_decoder": use_unified_decoder  # Whether to use the unified LLMDecoder class
    }
    
    # MERT configuration (only takes effect when saving configuration is not used)
    mert_config = {
        "model_size": "MERT_330m",  # MERT model size, consistent with training
        "layer_selection": "all"  # MERT layer selection, supported formats: "all", "12", "1,5,10,12"
    }
    
    # Only print hardcoded config if not using saved config
    if not use_saved_config:
        print("Using the latest OnsetsAndFrames architecture: Unifying feature extraction and decoding")
        print(f"Encoder model: {encoder_model}")
        print(f"Encoder layer selection: {encoder_layer_selection}")
        print(f"Decoder model: {decoder_model}")
        print(f"Decoder layer selection: {decoder_layer_selection}")
        print(f"Use unityLLMDecoder: {use_unified_decoder}")
        print(f"Decoder freezes: {decoder_freeze_model}")
        print(f"Head mode: {head_mode}")
    else:
        print("The configuration saved during training will be used...")
    
    # Perform transcription in gradient-free mode to improve inference efficiency
    with torch.no_grad():
        transcribe_file(model_file, audio_paths, save_path, sequence_length,
                       onset_threshold, frame_threshold, segment_duration, device, use_saved_config, tempo,
                       audio_root=audio_root)


if __name__ == '__main__':
    # Create a command line parameter parser - only keep three necessary parameters
    parser = argparse.ArgumentParser(description='Music Transcription Script - Automatically reads configuration from training records')
    
    #Basic parameters - only keep three necessary parameters
    parser.add_argument('--training_record_dir', type=str, default=TRAINING_RECORD_DIR,
                        help='Training record folder path (including model-best-note.pt and 1/config.json)')
    parser.add_argument('--audio_path', type=str, nargs='+', default=AUDIO_PATHS,
                        help='Enter audio file path (supports file, folder or wildcard mode, multiple can be specified)')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                        help='Result saving directory path')
    parser.add_argument('--audio_root', type=str, default=None,
                        help='Root directory of input audio; preserves subfolder layout in save_path')
    
    #Device parameters
    parser.add_argument('--device', type=str, default=None,
                        help='Computing device (such as cuda:0, cuda:1, cpu). If not specified, the available GPU will be automatically detected, and if there is no GPU, the cpu will be used.')
    
    args = parser.parse_args()
    
    # Process device parameters
    if args.device is None:
        if torch.cuda.is_available():
            # Automatically select the first available GPU
            args.device = 'cuda:0'
            print(f"automatically detectedGPU, use the device: {args.device}")
        else:
            args.device = 'cpu'
            print("GPU not detected, using CPU device")
    else:
        # Verify device parameters
        if args.device.startswith('cuda:'):
            gpu_id = int(args.device.split(':')[1])
            if gpu_id >= torch.cuda.device_count():
                print(f"Error: specifiedGPU {args.device} does not exist, only {torch.cuda.device_count()} indivualGPUAvailable")
                print(f"availableGPUequipment: {[f'cuda:{i}' for i in range(torch.cuda.device_count())]}")
                sys.exit(1)
        elif args.device != 'cpu':
            print(f"Warning: Unknown device parameters '{args.device}', will use automatic detection")
            args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"Use equipment: {args.device}")
    
    # Disable gradient calculation (inference phase, save video memory and accelerate)
    with torch.no_grad():
        # Call the transcription function
        main(args.training_record_dir, args.audio_path, args.save_path, args.device, SEGMENT_DURATION,
             args.audio_root)
