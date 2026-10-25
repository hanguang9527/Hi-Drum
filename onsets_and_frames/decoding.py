import numpy as np
import torch


def extract_notes(onsets, frames, velocity, onset_threshold=0.5, frame_threshold=0.5):
    """
    Finds the note timings based on the onsets and frames information

    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float

    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    velocities: np.ndarray of velocity values
    """
    # GPUOptimization: Usetorch.no_grad()Reduce memory usage
    with torch.no_grad():
        device = onsets.device
        
        # GPUOptimization: keep tensors inGPUPerform threshold processing on the system and use more efficient data types
        onsets_binary = (onsets > onset_threshold).to(torch.bool)
        frames_binary = (frames > frame_threshold).to(torch.bool)
        
        # GPUOptimization: inGPUCalculationonsetDifferential, using more efficient operations
        onset_diff = torch.zeros_like(onsets_binary)
        onset_diff[1:] = onsets_binary[1:] & (~onsets_binary[:-1])
        onset_diff[0] = onsets_binary[0]

        # GPUOptimization: get all in batchesonsetLocation
        onset_positions = onset_diff.nonzero()
        
        if onset_positions.numel() == 0:
            return np.array([]), np.array([]).reshape(0, 2), np.array([])

        pitches = []
        intervals = []
        velocities = []

        # GPUOptimization: reduceCPUConversion, batch processing
        n_frames, n_bins = onsets_binary.shape
        
        # Convert only necessary data toCPU
        onset_positions_cpu = onset_positions.cpu().numpy()
        onsets_cpu = onsets_binary.cpu().numpy()
        frames_cpu = frames_binary.cpu().numpy()
        velocity_cpu = velocity.cpu().numpy()

        for pos in onset_positions_cpu:
            frame, pitch = pos[0], pos[1]
            
            onset = frame
            offset = frame
            velocity_samples = []

            # Optimization: UsenumpyVectorized operation search foroffset
            pitch_column = onsets_cpu[frame:, pitch] | frames_cpu[frame:, pitch]
            if pitch_column.any():
                # find the last oneTruelocation
                last_true = np.where(pitch_column)[0]
                if len(last_true) > 0:
                    offset = frame + last_true[-1] + 1
                    
                    # collectvelocitysample
                    onset_mask = onsets_cpu[frame:offset, pitch]
                    if onset_mask.any():
                        velocity_samples = velocity_cpu[frame:offset, pitch][onset_mask]

            if offset > onset:
                pitches.append(pitch)
                intervals.append([onset, offset])
                velocities.append(np.mean(velocity_samples) if len(velocity_samples) > 0 else 0)

    return np.array(pitches), np.array(intervals), np.array(velocities)


def notes_to_frames(pitches, intervals, shape):
    """
    Takes lists specifying notes sequences and return

    Parameters
    ----------
    pitches: list of pitch bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]

    Returns
    -------
    time: np.ndarray containing the frame indices
    freqs: list of np.ndarray, each containing the frequency bin indices
    """
    roll = np.zeros(tuple(shape))
    for pitch, (onset, offset) in zip(pitches, intervals):
        roll[onset:offset, pitch] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return time, freqs


def notes_to_frames_gpu(pitches, intervals, shape, device=None):
    """
    GPUOptimized version: convert note information into frame representation
    
    Parameters
    ----------
    pitches: np.ndarray or torch.Tensor of pitch bin indices
    intervals: np.ndarray or torch.Tensor of [onset, offset] ranges
    shape: tuple, the shape of the original piano roll, [n_frames, n_bins]
    device: torch.device, GPUequipment
    
    Returns
    -------
    time: torch.Tensor containing the frame indices
    freqs: list of torch.Tensor, each containing the frequency bin indices
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPUOptimization function, supportGPUaccelerate
    
    with torch.no_grad():
        # Make sure the input istorchTensor
        if isinstance(pitches, np.ndarray):
            pitches = torch.from_numpy(pitches).to(device)
        if isinstance(intervals, np.ndarray):
            intervals = torch.from_numpy(intervals).to(device)
        
        # createGPUonpiano roll
        roll = torch.zeros(shape, dtype=torch.bool, device=device)
        
        # Batch fillpiano roll
        if len(pitches) > 0:
            for i, (pitch, interval) in enumerate(zip(pitches, intervals)):
                onset, offset = interval[0].item(), interval[1].item()
                if offset > onset and onset < shape[0] and pitch < shape[1]:
                    roll[onset:min(offset, shape[0]), pitch] = True
        
        # Generate time index
        time = torch.arange(roll.shape[0], device=device)
        
        # Get the frequency index of each time frame in batches
        freqs = []
        for t in range(roll.shape[0]):
            freq_indices = roll[t, :].nonzero().squeeze(-1)
            freqs.append(freq_indices)
    
    return time, freqs


def extract_notes_batch_gpu(onsets_batch, frames_batch, velocity_batch, 
                           onset_threshold=0.5, frame_threshold=0.5):
    """
    batchGPUOptimized version of note extraction function
    
    Parameters
    ----------
    onsets_batch: torch.Tensor, shape = [batch, frames, bins]
    frames_batch: torch.Tensor, shape = [batch, frames, bins]
    velocity_batch: torch.Tensor, shape = [batch, frames, bins]
    onset_threshold: float
    frame_threshold: float
    
    Returns
    -------
    batch_results: list of tuples (pitches, intervals, velocities) for each batch item
    """
    batch_results = []
    
    with torch.no_grad():
        batch_size = onsets_batch.shape[0]
        
        for b in range(batch_size):
            pitches, intervals, velocities = extract_notes(
                onsets_batch[b], frames_batch[b], velocity_batch[b],
                onset_threshold, frame_threshold
            )
            batch_results.append((pitches, intervals, velocities))
    
    return batch_results
