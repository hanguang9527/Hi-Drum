# Import operating system interface library (for file path processing)
import os
# fromabcModule imports abstract method decorator (used to define abstract base classes)
from abc import abstractmethod
from collections import defaultdict
# fromglobmodule importglobFunction (for matching file paths)
from glob import glob

# importnumpyLibrary (for numerical calculations)
import numpy as np
# importsoundfileLibrary (for reading and writing audio files)
import soundfile
# fromtorch.utils.dataimportDatasetkind(PyTorchDataset base class)
from torch.utils.data import Dataset
# importPyTorchLibrary (deep learning framework)
import torch
# importtqdmLibrary (for displaying progress bars)
from tqdm import tqdm

# importh5pylibrary (forHDF5File reading and writing, efficient storage of large data)
import h5py

# Import constants from the current package (such as sample rate、Jump length, etc., are not defined in the code and need to be combined with external files)
from .constants import *

# DisableHDF5File locking (to avoid file locking problems when loading multiple processes)
os.environ['HDF5_USE_FILE_LOCKING'] = "FALSE"


class PianoRollAudioDataset(Dataset):
    """
    Piano Roll (Piano Roll) and audio data set base class, inherited fromPyTorchofDataset
    Used to load audio and correspondingMIDItag data (onset、offset、pitch、intensity, etc.)
    """
    def __init__(self, path, groups=None, sequence_length=None, seed=42):
        """
        Initialize the data set
        :param path: Dataset root directory path
        :param groups: Data set grouping (e.g.train/test/validation), the default isNoneuse all available groupings
        :param sequence_length: Audio sequence length (used to intercept fixed-length segments),NoneIndicates using full audio
        :param seed: Random number seed (used for random starting positions when intercepting fragments)
        """
        self.path = path  # Dataset path
        # Determine which grouping to use (if not specified all available groups are used)
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length  # audio sequence length
        self.random = np.random.RandomState(seed)  # Random number generator (fixed seed ensures reproducibility)

        self.data = []  # List of storage data file paths
        
        # Print loading information
        print(f"Loading {len(self.groups)} group{'s' if len(self.groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {self.path}")
        
        # Traverse each group and load the data file path
        for group in self.groups:
            # Iterate through all file pairs within the group (audio files+tag file), usetqdmshow progress
            for input_files in tqdm(self.files(group), desc=f'Loading group {group}'):
                # callloadmethod processes the file and returns the result (HDF5file path) added to the data list
                self.data.append(self.load(*input_files))
        # Why: 训练按固定 sequence_length 切段后直接 stack。短于该长度的音频切片更短，
        # 一个 batch 里长度不一致就会在 collate 崩掉。这里在建索引时丢掉它们，5 秒协议下
        # 现有片段都够长，行为不变；10 秒实验才会真正滤掉大约一成。
        self._drop_clips_shorter_than_sequence()
        # Why: grouped multi-kit training needs to reuse the same cached HDF5 sample
        # lengths across many synchronized reads; memoizing avoids repeated metadata I/O.
        self._audio_lengths = {}



    def _read_h5_sample(self, h5_path, sample_begin=None):
        '''
        Get the sample at the specified index
        Return value format:
        [n_step, midi_bins]  # n_stepis the number of time steps,midi_binsforMIDInumber of pitches
        '''
        # OpenHDF5File reading data
        with h5py.File(h5_path, 'r') as data:
            # Store audio file path (used to trace the original data)
            result = dict(path=data['path'][()])
            step_begin = 0
            step_end = data['label'].shape[0]
            begin = 0
            end = data['audio'].shape[0]

            # If the sequence length is specified, a fixed-length fragment is truncated
            if self.sequence_length is not None:
                # Get the total audio length (number of samples)
                audio_length = data['audio'].shape[0]
                # Calculate a random starting time step (to ensure the fragment does not cross boundaries)
                max_start = max(0, audio_length - self.sequence_length)
                if max_start == 0:
                    step_begin = 0
                elif sample_begin is not None:
                    # Why: Groove-EGMD grouped batches should observe the same temporal
                    # window across kits, so external callers may pin a shared sample start.
                    step_begin = min(sample_begin, max_start) // HOP_LENGTH
                else:
                    step_begin = self.random.randint(max_start) // HOP_LENGTH
                # Calculate the number of time steps corresponding to the sequence length (HOP_LENGTHStep length for audio feature extraction)
                n_steps = self.sequence_length // HOP_LENGTH
                step_end = step_begin + n_steps  # end time step

                # Calculate the start and end index of an audio sample
                begin = step_begin * HOP_LENGTH
                end = begin + self.sequence_length

            # Why: 统一在这里做一次切片读取，避免前面先读进 result、后面又重复从 HDF5
            # 取同一段数据，减少一次无意义的 I/O 和中间数组创建。
            audio_data = data['audio'][begin:end]
            label_data = data['label'][step_begin:step_end, :]
            velocity_data = data['velocity'][step_begin:step_end, :]

            # Why: 当 sequence_length=None 时，数据集语义是“读取完整音频”；
            # 这里改用实际片段长度做归一化，避免 None / 10 直接崩溃，同时保持和推理侧按片段长度归一化的行为一致。
            normalization_length = self.sequence_length if self.sequence_length is not None else len(audio_data)
            result['audio'] = torch.tensor(audio_data, dtype=torch.float32).div_(normalization_length / 10)
            
            result['label'] = torch.tensor(label_data, dtype=torch.float32)
            result['velocity'] = torch.tensor(velocity_data, dtype=torch.float32).div_(128.0)
            
            # GPUOptimization: Batch calculation of all tag types
            result['onset'] = (result['label'] == 3).float()
            # result['offset'] = (result['label'] == 1).float()  # 当前模型已移除对offset（音符结束时间）的预测和Loss计算（因为drum鼓不需要计算offset），将其注释掉以节省CPU计算和内存开销。
            result['frame'] = (result['label'] > 1).float()

            # Downsampling (currently commented out, uncommented if required)
            # result['onset'] = max_pooling(result['onset'], [2, 1])
            # result['offset'] = max_pooling(result['offset'], [2, 1])
            # result['frame'] = max_pooling(result['frame'], [2, 1])
            # result['velocity'] = max_pooling(result['velocity'], [2, 1])

        return result

    def __getitem__(self, index):
        # Get the current sampleHDF5file path
        h5_path = self.data[index]
        return self._read_h5_sample(h5_path)

    def get_item(self, index, sample_begin=None):
        """
        Return a sample by index, optionally forcing a shared audio crop start.
        """
        return self._read_h5_sample(self.data[index], sample_begin=sample_begin)

    def _drop_clips_shorter_than_sequence(self):
        if self.sequence_length is None or not self.data:
            return
        kept = []
        dropped = 0
        for h5_path in self.data:
            with h5py.File(h5_path, 'r') as data:
                audio_length = int(data['audio'].shape[0])
            if audio_length >= self.sequence_length:
                kept.append(h5_path)
            else:
                dropped += 1
        if dropped:
            print(
                f'Dropped {dropped}/{dropped + len(kept)} clips shorter than '
                f'{self.sequence_length} samples '
                f'({self.sequence_length / SAMPLE_RATE:.1f}s) in {self.groups}'
            )
        if not kept:
            raise ValueError(
                f'No clips remain in {self.groups} after dropping audio shorter than '
                f'{self.sequence_length} samples.'
            )
        self.data = kept

    def get_audio_length(self, index):
        """
        Return the cached audio length for the indexed HDF5 sample.
        """
        if index not in self._audio_lengths:
            with h5py.File(self.data[index], 'r') as data:
                self._audio_lengths[index] = data['audio'].shape[0]
        return self._audio_lengths[index]

    def __len__(self):
        """Returns the number of samples in the dataset"""
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """Abstract method: Returns all available group names in the current data set"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """Abstract method: Returns all file pairs within the specified group (audio file path, label file path)"""
        raise NotImplementedError

    def load(self, audio_path, tsv_path):
        """
        Load audio files and corresponding tag files and convert toHDF5Format and save (avoid duplication of processing)
        returnHDF5file path

        The return value contains:
        path: audio file path
        audio: Original waveform (torch.ShortTensor,shape[num_samples])
        label: label matrix(torch.ByteTensor,shape[num_steps, midi_bins])
               Encoding rules:3=onset,2=onsetAfter the continuous frame,1=offset,0=No events
        velocity: velocity matrix (torch.ByteTensor,shape[num_steps, midi_bins]),storageMIDIvelocity value
        """
        # structureHDF5File save path
        # Separate the root directory and file name of an audio file
        root, basename = os.path.split(audio_path)
        # Separate filename and extension
        name, ext = os.path.splitext(basename)
        # definitionHDF5The file storage directory (created one level above the audio directoryh5_hop{Jump length}folder)
        h5_dir = os.path.join(root, '..', f'h5_hop{HOP_LENGTH}')
        # Create directory if it does not exist
        os.makedirs(h5_dir, exist_ok=True)
        # wholeHDF5file path
        h5_path = os.path.join(h5_dir, name + '.h5')

        # ifHDF5If the file already exists, the path will be returned directly (no need to repeat processing)
        if os.path.exists(h5_path):
            return h5_path

        # Otherwise, process the audio and tags and save asHDF5document
        with h5py.File(h5_path, mode='w') as h5:
            # Read an audio file (return audio data and sample rate),dtype='int16'express16bit integer format
            audio, sr = soundfile.read(audio_path, dtype='int16')
            # The following is an alternate audio loading method (usinglibrosa), the current annotation is not enabled
            # audio, sr = librosa.load(audio_path, sr = SAMPLE_RATE, mono=True)
            # assert sr == SAMPLE_RATE  # Make sure the sampling rate is as expected

            # Convert audio data toShortTensor
            audio = torch.ShortTensor(audio)
            # Total audio length (number of samples)
            audio_length = len(audio)

            # calculateMIDInumber of pitches (fromMIN_MIDIarriveMAX_MIDIrange)
            n_keys = MAX_MIDI - MIN_MIDI + 1
            # Calculate the number of time steps (based on audio length and skip length)
            n_steps = (audio_length - 1) // HOP_LENGTH + 1

            # Initialize the label matrix and velocity matrix (full0)
            label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
            velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

            # If the tag file path is valid, load and process tags
            if (len(tsv_path) > 0):
                # fromTSVFile loadingMIDIevent data (onsettime、offsettime、pitch、intensity)
                midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)  # skiprows=1skip header
                # Why: np.loadtxt 在单条事件标签上会返回一维数组；统一升成二维后，
                # 下面的事件循环就不会因为“单行 tsv”而解包崩溃。
                if midi.size == 0:
                    midi = np.empty((0, 4))
                else:
                    midi = np.atleast_2d(midi)

                # Iterate through eachMIDIevent
                for onset, offset, note, vel in midi:
                    # calculateonsetThe corresponding time step (onsettime(Second) * Sampling rate / Skip length, rounded)
                    left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
                    # onsetduration of time steps (HOPS_IN_ONSETfor predefinedonsetcontinuous step size)
                    onset_right = min(n_steps, left + HOPS_IN_ONSET)
                    # calculateoffsetThe corresponding time step
                    frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
                    frame_right = min(n_steps, frame_right)  # Make sure not to exceed the total step size
                    # offsetduration of time steps (HOPS_IN_OFFSETfor predefinedoffsetcontinuous step size)
                    offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

                    # Calculate the index of the current pitch in the matrix (relative toMIN_MIDIoffset)
                    f = int(note) - MIN_MIDI
                    # markonset(3)、Continuous Frame (2)、offset(1)
                    label[left:onset_right, f] = 3
                    label[onset_right:frame_right, f] = 2
                    label[frame_right:offset_right, f] = 1
                    # Record the velocity value (inonsetarriveoffsetframes between)
                    velocity[left:frame_right, f] = vel

            # write dataHDF5file (adaptationdataset1.pyData structure, not pre-storedMERTfeature)
            h5['path'] = audio_path  # Store original audio path
            h5['audio'] = audio.numpy()  # Store audio data (converted tonumpyarray)
            h5['label'] = label.numpy()  # Store label matrix
            h5['velocity'] = velocity.numpy()  # Store velocity matrix

        # returnHDF5file path
        return h5_path


class Groove(PianoRollAudioDataset):
    """
    GrooveDataset class for processinggroove_processed_10s_24khzDataset
    
    This dataset contains preprocessed10seconds audio clip, sample rate24kHz
    Data organization structure:
    - train/: Training set audio files
    - validation/: Verification set audio file  
    - test/: Test set audio files
    - labels/: The corresponding label file (TSVFormat)
    """
    def __init__(self, path='data/groove_processed_10s_24kh7class', groups=None, sequence_length=None, seed=42):
        """
        initializationGrooveDataset
        
        parameter:
        path: Data set root path, default is'data/groove_processed_10s_24khz'
        groups: Data grouping list, default is['train', 'validation', 'test']
        sequence_length: Sequence length (number of time steps), used to intercept fixed-length audio clips
        seed: Random seeds for data shuffling and sampling repeatability
        """
        super().__init__(path, groups if groups is not None else ['train', 'validation', 'test'], sequence_length, seed)    

    @classmethod
    def available_groups(cls):
        """
        returnGrooveAll available grouping names for the dataset
        
        return:
        list: List containing all available group names
        """
        return ['train', 'validation', 'test']

    def files(self, group):
        """
        Get all file pairs of the specified group (audio file path, label file path)
        
        parameter:
        group: Data group name ('train', 'validation', 'test'one)
        
        return:
        list: Include(audio file path, Label file path)list of tuples
        """
        if group not in self.available_groups():
            raise ValueError(f"Unrecognized group: {group}")

        # Construct audio file directory path
        audio_dir = os.path.join(self.path, f'{group}_data')
        # Construct label file directory path
        label_dir = os.path.join(self.path, f'{group}_label')
        
        # Check if directory exists
        if not os.path.exists(audio_dir):
            raise FileNotFoundError(f"Audio directory does not exist: {audio_dir}")
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"tag directory does not exist: {label_dir}")
        
        # Get a list of audio files (only handles.flacFormat)
        audio_files = []
        # only use.flacaudio file format
        audio_files.extend(glob(os.path.join(audio_dir, '*.flac')))
        
        # Sort all audio files uniformly (fix: avoid misalignment problems caused by separate sorting)
        audio_files = sorted(audio_files)
        
        # Construct a list of file pairs
        result = []
        for audio_path in audio_files:  # Fix: Use sorted audio file list directly
            # Get the base name of the audio file (without extension)
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            # Construct the corresponding label file path
            tsv_path = os.path.join(label_dir, basename + '.tsv')
            
            # Check if tag file exists
            if not os.path.exists(tsv_path):
                print(f"Warning: The corresponding tag file cannot be found: {tsv_path}")
                # If there is no label file, use an empty string as the label path (indicating unlabeled data)
                tsv_path = ""
            
            result.append((audio_path, tsv_path))
        
        return result


def _egmd_group_key(stem):
    """
    Normalize an E-GMD filename stem back to its Groove-aligned performance key.
    """
    if '_ext' in stem:
        prefix, suffix = stem.split('_ext', 1)
        head, sep, tail = prefix.rpartition('_')
        if sep and tail.isdigit():
            return f'{head}_ext{suffix}'
    head, sep, tail = stem.rpartition('_')
    if sep and tail.isdigit():
        return head
    return stem


class GroupedGrooveEgmdSlakhDataset(Dataset):
    """
    Training dataset that returns a ready-to-train mini-batch composed of:
    - one Groove anchor,
    - several E-GMD timbre variants with the same label,
    - optional Slakh samples for additional real-audio diversity.
    """

    def __init__(
        self,
        groove_path,
        egmd_path,
        slakh_path=None,
        groups=None,
        sequence_length=None,
        seed=42,
        matched_samples_per_batch=5,
        unmatched_samples_per_batch=2,
        slakh_samples_per_batch=1,
        total_batch_size=8,
    ):
        self.groups = groups if groups is not None else ['train']
        self.sequence_length = sequence_length
        self.random = np.random.RandomState(seed)
        self.matched_samples_per_batch = max(1, matched_samples_per_batch)
        self.unmatched_samples_per_batch = max(0, unmatched_samples_per_batch)
        self.slakh_samples_per_batch = max(0, slakh_samples_per_batch)
        self.total_batch_size = max(1, total_batch_size)
        if (
            self.matched_samples_per_batch
            + self.unmatched_samples_per_batch
            + self.slakh_samples_per_batch
            != self.total_batch_size
        ):
            raise ValueError(
                'matched_samples_per_batch + unmatched_samples_per_batch + '
                'slakh_samples_per_batch must equal total_batch_size.'
            )
        self.egmd_variants_per_anchor = max(0, self.matched_samples_per_batch - 1)

        self.groove = Groove(
            path=groove_path,
            groups=self.groups,
            sequence_length=sequence_length,
            seed=seed,
        )
        self.egmd = Groove(
            path=egmd_path,
            groups=self.groups,
            sequence_length=sequence_length,
            seed=seed + 1,
        )
        self.slakh = None
        if slakh_path:
            self.slakh = Groove(
                path=slakh_path,
                groups=self.groups,
                sequence_length=sequence_length,
                seed=seed + 2,
            )

        groove_lookup = {
            os.path.splitext(os.path.basename(h5_path))[0]: idx
            for idx, h5_path in enumerate(self.groove.data)
        }
        egmd_lookup = defaultdict(list)
        for idx, h5_path in enumerate(self.egmd.data):
            stem = os.path.splitext(os.path.basename(h5_path))[0]
            egmd_lookup[_egmd_group_key(stem)].append(idx)

        self.group_keys = sorted(
            key for key in groove_lookup.keys() if key in egmd_lookup and len(egmd_lookup[key]) > 0
        )
        self.group_to_indices = {
            key: {
                'groove': groove_lookup[key],
                'egmd': sorted(egmd_lookup[key]),
            }
            for key in self.group_keys
        }

        matched_egmd_indices = {
            local_index
            for mapping in self.group_to_indices.values()
            for local_index in mapping['egmd']
        }
        self.unmatched_groove_indices = sorted(
            groove_lookup[key]
            for key in groove_lookup.keys()
            if key not in self.group_to_indices
        )
        self.unmatched_egmd_indices = sorted(
            local_index
            for local_index in range(len(self.egmd))
            if local_index not in matched_egmd_indices
        )
        self.slakh_indices = list(range(len(self.slakh))) if self.slakh is not None else []
        if self.slakh_samples_per_batch > 0 and not self.slakh_indices:
            raise ValueError('slakh_samples_per_batch > 0 but no Slakh dataset was provided.')

        # Why: unmatched Groove and unmatched E-GMD represent different kinds of
        # coverage holes, so split the unmatched quota across two dedicated pools.
        groove_quota = self.unmatched_samples_per_batch // 2
        egmd_quota = self.unmatched_samples_per_batch - groove_quota
        if self.unmatched_samples_per_batch > 0:
            if not self.unmatched_groove_indices:
                egmd_quota = self.unmatched_samples_per_batch
                groove_quota = 0
            elif not self.unmatched_egmd_indices:
                groove_quota = self.unmatched_samples_per_batch
                egmd_quota = 0
            elif groove_quota == 0:
                groove_quota = 1
                egmd_quota = self.unmatched_samples_per_batch - 1
        self.unmatched_groove_quota = groove_quota
        self.unmatched_egmd_quota = egmd_quota

        self._pool_indices = {}
        self._pool_positions = {}
        self._reset_pool('unmatched_groove', self.unmatched_groove_indices)
        self._reset_pool('unmatched_egmd', self.unmatched_egmd_indices)
        self._reset_pool('slakh', self.slakh_indices)

        if not self.group_keys:
            raise ValueError('No matched Groove/E-GMD groups were found for grouped training.')

    def __len__(self):
        return len(self.group_keys)

    def _reset_pool(self, pool_name, source_indices):
        shuffled = list(source_indices)
        if shuffled:
            self.random.shuffle(shuffled)
        self._pool_indices[pool_name] = shuffled
        self._pool_positions[pool_name] = 0

    def _draw_from_pool(self, pool_name, source_indices, count):
        if count <= 0 or not source_indices:
            return []
        drawn = []
        while len(drawn) < count:
            if self._pool_positions[pool_name] >= len(self._pool_indices[pool_name]):
                self._reset_pool(pool_name, source_indices)
            if not self._pool_indices[pool_name]:
                break
            next_index = self._pool_indices[pool_name][self._pool_positions[pool_name]]
            self._pool_positions[pool_name] += 1
            drawn.append(next_index)
        return drawn

    def _choose_shared_sample_begin(self, selected_members):
        if self.sequence_length is None:
            return None
        max_starts = []
        for dataset, local_index in selected_members:
            audio_length = dataset.get_audio_length(local_index)
            max_starts.append(max(0, audio_length - self.sequence_length))
        shared_max_start = min(max_starts) if max_starts else 0
        if shared_max_start == 0:
            return 0
        return self.random.randint(shared_max_start)

    def _stack_samples(self, samples):
        batch = {
            # Why: keeping the original paths in the grouped batch makes it easier to
            # debug which timbre variants and Slakh examples were drawn for a step.
            'path': [sample['path'] for sample in samples],
            'audio': torch.stack([sample['audio'] for sample in samples], dim=0),
            'label': torch.stack([sample['label'] for sample in samples], dim=0),
            'velocity': torch.stack([sample['velocity'] for sample in samples], dim=0),
            'onset': torch.stack([sample['onset'] for sample in samples], dim=0),
            'frame': torch.stack([sample['frame'] for sample in samples], dim=0),
        }
        return batch

    def __getitem__(self, index):
        group_key = self.group_keys[index]
        mapping = self.group_to_indices[group_key]

        egmd_candidates = mapping['egmd']
        egmd_count = self.egmd_variants_per_anchor
        if egmd_count > 0 and egmd_candidates:
            # Why: some matched groups have fewer rendered kit variants than the
            # target quota; replacement keeps the matched portion size stable.
            chosen_positions = self.random.choice(
                len(egmd_candidates),
                size=egmd_count,
                replace=len(egmd_candidates) < egmd_count,
            )
            chosen_egmd = [egmd_candidates[int(pos)] for pos in np.atleast_1d(chosen_positions)]
        else:
            chosen_egmd = []

        selected_group_members = [(self.groove, mapping['groove'])]
        selected_group_members.extend((self.egmd, local_index) for local_index in chosen_egmd)
        shared_sample_begin = self._choose_shared_sample_begin(selected_group_members)

        samples = [
            dataset.get_item(local_index, sample_begin=shared_sample_begin)
            for dataset, local_index in selected_group_members
        ]

        for local_index in self._draw_from_pool(
            'unmatched_groove',
            self.unmatched_groove_indices,
            self.unmatched_groove_quota,
        ):
            samples.append(self.groove.get_item(local_index))

        for local_index in self._draw_from_pool(
            'unmatched_egmd',
            self.unmatched_egmd_indices,
            self.unmatched_egmd_quota,
        ):
            samples.append(self.egmd.get_item(local_index))

        for local_index in self._draw_from_pool(
            'slakh',
            self.slakh_indices,
            self.slakh_samples_per_batch,
        ):
            samples.append(self.slakh.get_item(local_index))

        return self._stack_samples(samples)
