import os
# Cap host thread pools before numpy/torch import so multi-GPU jobs don't
# each spawn ~nproc/2 CPU threads and oversubscribe the machine.
for _thread_env in (
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'TORCH_NUM_THREADS',
):
    os.environ.setdefault(_thread_env, '4')

from datetime import datetime
import numpy as np
import inspect
import json
import torch
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_

_torch_threads = max(1, int(os.environ.get('TORCH_NUM_THREADS', '4')))
torch.set_num_threads(_torch_threads)
try:
    torch.set_num_interop_threads(min(_torch_threads, 4))
except RuntimeError:
    pass
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate
from onsets_and_frames import *
from onsets_and_frames.transcriber import run_transcription_batch
from onsets_and_frames.batch_config import apply_gpu_memory_batch_defaults
from onsets_and_frames.checkpointing import (
    OPTIMIZER_LATEST,
    SCHEDULER_LATEST,
    build_train_meta,
    decide_resume_restore,
    find_model_weights_path,
    load_best_metrics_from_txt,
    load_checkpoint_meta,
    resolve_checkpoint_dir,
    resolve_resume_iteration,
    save_training_checkpoint,
)
from onsets_and_frames.constants import DEFAULT_LAYER_SELECTION, SAMPLE_RATE
from onsets_and_frames.distributed import (
    barrier_if_distributed,
    broadcast_object,
    cleanup_distributed,
    init_distributed,
    is_main_process,
)

ex = Experiment('train_transcriber', save_git_info=False)  # Disable git probing; large untracked trees make Sacred repo.is_dirty() hang


@ex.config
def config():
    distributed, rank, local_rank, world_size = init_distributed()
    if distributed:
        device = f'cuda:{local_rank}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')

    train_on = 'GrooveEgmd'
    dataset_path = './data/groove_processed_5_10_24khz_7class'
    egmd_dataset_path = './data/egmd_processed_5_10_24khz_7class'
    slakh_dataset_path = ''
    matched_samples_per_batch = 5
    unmatched_samples_per_batch = 3
    slakh_samples_per_batch = 0
    validation_include_egmd = True
    
    # ========== Model selection and layer selection parameter configuration area ==========
    # 🎯 encoder selection parameters (new): select feature extraction source
    # Optional: 'MERT_95m', 'MERT_330m', 'MERT2' (MERT-v2-30s, 24 layers, 25 Hz), 'MusicFM', 'MuQ'
    encoder_model = 'MERT_330m'
    
    # 🎯 Encoder layer selection parameters (used for all feature extraction models)
    # Information about the number of layers of each model:
    # - MERT_95m: 13 layers (0-12)
    # - MERT_330m: 25 floors (0-24)
    # - MERT2: 24 blocks (0-23). hidden_states[0] is block 1.
    # - MusicFM: Layer 13 (Layer 12 Conformer + Layer 1 interface, index 0-12)
    # - MuQ: (12-layer Conformer + 1-layer interface, index 0-12) Pay special attention here to the 12-layer Conformer written in the paper
    #Supported layer selection formats:
    # - "all": Use all layers for feature fusion (default)
    # - Single layer: "12" (only use layer 12)
    # - Multi-layer: "0,6,12" (use layers 0, 6, and 12 for multi-layer fusion)
    encoder_layer_selection = DEFAULT_LAYER_SELECTION  # Default is "all"
    
    # 🎯 Decoder layer selection parameters (used for feature fusion of all decoder models)
    # Information on the number of layers of each decoder model:
    # - Qwen3-0.6B: 28 layers (0-27)
    # - GLM-0.5B: 24 layers (0-23)
    # - Hunyuan-0.5B-Instruct: 24 layers (0-23)
    # - MiniCPM4-0.5B: 24 layers (0-23)
    # - Falcon-H1-0.5B-Base: 36 layers (0-35)
    #Supported layer selection formats (consistent with MERT layer selection mechanism):
    # - Single layer: "23" (only use layer 23)
    # - Multi-layer: "1,5,10,15,20,23" (use 6 evenly distributed layers)
    # - Fewer layers: "5,15,23" (using 3 key layers, less computational overhead)
    # - More layers: "0,2,4,6,8,10,12,14,16,18,20,22" (use 12 even layers)
    # Note: The layer number starts counting from 0. Please adjust the layer number range according to the specific model.
    decoder_layer_selection = "all"
    # 🎯 Decoder model configuration (when using the unified LLMDecoder class)
    #Supported decoder models (when use_unified_decoder = True):

    # - "Qwen2.5-0.5B": legacy configuration, it is recommended to use "Qwen3-0.6B"
    # - "Qwen3-0.6B": Qwen/Qwen3-0.6B (28 layers, 1024 dimensions)
    # - "MiniCPM4-0.5B": openbmb/MiniCPM4-0.5B (24 layers, 1024 dimensions)
    # - "Hunyuan-0.5B-Instruct": Tencent-Hunyuan/HunyuanDenseV1-0.5B-Instruct (24 layers, 1024 dimensions)
    # - "GLM-0.5B": THUDM/GLM-4-9B-Chat (24 layers, 1024 dimensions) Note that this has not been tested with musicFM yet
    # - "Falcon-H1-0.5B-Base": tiiuae/Falcon-H1-0.5B-Base (36 layers, 1024 dimensions) Note that this has not been tested with musicFM yet

    decoder_model = "Qwen3-0.6B"
    
    decoder_freeze_model = True  # Whether to freeze decoder model parameters
    # ICASSP B1: llm | scratch_transformer | none
    decoder_backend = "llm"
    # ICASSP encoder FT axis: False = fine-tune encoder (historical default)
    encoder_freeze = False
    
    # 🧪 Ablation experiment configuration (Adapter parallel branch control)
    #Control the enabling status of three parallel branches in the Adapter
    # Only when input_dim == output_dim (such as MERT 330m + Qwen3 0.6B) supports full False (0 branch/direct connection)
    enable_branch1 = True  # Whether to enable parallel convolution branch (ParallelConv)
    enable_branch2 = True  # Whether to enable dilated convolution branch (DilatedConv)
    enable_branch3 = True  # Whether to enable the Transformer branch (Transformer)

    # 🎯 Multi-task heads configuration
    # - "independent": fully separate onset/frame/velocity heads
    # - "paper_hierarchical": shared global conv; onset conditions frame and velocity
    head_mode = 'paper_hierarchical'

    # 🎯 Multi-task loss weights (onset : frame : velocity)
    # Default 2:1:1 matches the historical training recipe.
    loss_onset_weight = 2
    loss_frame_weight = 1
    loss_velocity_weight = 1

    # ================================================  
    
    # Build a log directory name containing complete model configuration information
    if distributed:
        gpu_id = f'ddp{world_size}-rank{local_rank}'
    else:
        gpu_id = cuda_visible_devices.replace(',', '-') if cuda_visible_devices != 'not set' else 'cpu'
    
    # Handle encoder layer selection parameters for directory names: replace commas with underscores, "all" remains unchanged
    encoder_layer_str = encoder_layer_selection.replace(',', '_') if encoder_layer_selection != 'all' else 'all'
    
    # Handle decoder layer selection parameters for directory names: replace commas with underscores, "all" remains unchanged
    decoder_layer_str = decoder_layer_selection.replace(',', '_') if decoder_layer_selection != 'all' else 'all'
    
    # Extract the basename of the data set path for the directory name
    import os
    if train_on == 'GroupedGrooveEgmdSlakh':
        dataset_basename = 'grouped_groove_egmd_7class'
    elif train_on == 'GrooveEgmd':
        dataset_basename = 'concat_groove_egmd_7class'
    elif train_on == 'Egmd':
        dataset_basename = os.path.basename(egmd_dataset_path.rstrip('/')) if egmd_dataset_path else 'egmd_7class'
    else:
        dataset_basename = os.path.basename(dataset_path.rstrip('/')) if dataset_path else 'unknown_dataset'
    
    # Get the frequency information corresponding to HOP_LENGTH (25Hz) from constants.py
    from onsets_and_frames.constants import HOP_LENGTH, SAMPLE_RATE
    hop_freq = SAMPLE_RATE // HOP_LENGTH  # Calculate frequency (Hz)
    
    # Generate branch configuration string (such as TTT, TFT)
    # Sequence: branch1 (ParallelConv), branch2 (DilatedConv), branch3 (Transformer)
    branch_config_str = "".join(["T" if b else "F" for b in [enable_branch1, enable_branch2, enable_branch3]])
    
    # Generate heads configuration string for experiment naming
    head_mode_str = "HeadsPaper" if head_mode == "paper_hierarchical" else "HeadsInd"
    loss_weight_str = (
        f'w{loss_onset_weight}_{loss_frame_weight}_{loss_velocity_weight}'
    )
    # Compact freeze / backend tags for B1 runs
    backend_tag = {
        'llm': 'llm',
        'scratch_transformer': 'scratch',
        'none': 'none',
    }.get(str(decoder_backend).lower(), str(decoder_backend))
    enc_tag = 'encFZ' if encoder_freeze else 'encFT'
    dec_tag = 'decFZ' if decoder_freeze_model else 'decFT'
    
    # Build an enhanced log directory name, including all key parameter information
      # Format: encoder{encoder_model}_layer{encoder_layer_str}_decoder{decoder_model}_layer{decoder_layer_str}_dataset{dataset_basename}_hop{hop_freq}hz_{branch_config_str}_{post_proc_str}_gpu{gpu_id}_timestamp
    logdir_prefix = (
        f'runs/encoder_{encoder_model}_layer{encoder_layer_str}_decoder_{decoder_model}_'
        f'{backend_tag}_{enc_tag}_{dec_tag}_layer{decoder_layer_str}_dataset_{dataset_basename}_'
        f'hop{hop_freq}hz_{branch_config_str}_{head_mode_str}_{loss_weight_str}_gpu_{gpu_id}-'
    )
    if is_main_process(rank):
        logdir = logdir_prefix + datetime.now().strftime('%y%m%d-%H%M%S')
    else:
        logdir = None
    logdir = broadcast_object(logdir, src=0, distributed=distributed)

    iterations = 1000000
    # Resume:
    # - In-place:  `logdir=runs/<old> resume_iteration=auto`
    # - Fork:      `resume_from=runs/<old> resume_iteration=auto` (+ optional new knobs;
    #              writes into the newly created logdir)
    # - Legacy:    `logdir=runs/<old> resume_iteration=2000`
    # Changing world_size / batch rebuilds Adam; changing LR rebuilds the schedule.
    resume_from = None
    resume_iteration = None  # None | 'auto' | int
    checkpoint_interval = 2000

    batch_size = 8


    sequence_length = 120000
    model_complexity = 48

    if torch.cuda.is_available():
        gpu_index = local_rank if distributed else torch.cuda.current_device()
        total_gpu_memory = torch.cuda.get_device_properties(gpu_index).total_memory
        (
            batch_size,
            matched_samples_per_batch,
            unmatched_samples_per_batch,
            slakh_samples_per_batch,
            sequence_length,
            memory_message,
        ) = apply_gpu_memory_batch_defaults(
            total_gpu_memory,
            train_on,
            encoder_model,
            decoder_model,
            batch_size,
            matched_samples_per_batch,
            unmatched_samples_per_batch,
            slakh_samples_per_batch,
            sequence_length,
            distributed=distributed,
        )
        if memory_message and is_main_process(rank):
            print(memory_message)

    learning_rate = 0.0006      # Can be adjusted to 0.0003 later
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.90

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = sequence_length
    validation_interval = 2000

    if is_main_process(rank):
        ex.observers.append(FileStorageObserver.create(logdir))


@ex.automain
def train(logdir, device, iterations, resume_iteration, resume_from, checkpoint_interval, train_on, dataset_path, egmd_dataset_path,
          slakh_dataset_path, matched_samples_per_batch, unmatched_samples_per_batch, slakh_samples_per_batch,
          batch_size, sequence_length,
          model_complexity, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          clip_gradient_norm, validation_length, validation_interval, 
          encoder_layer_selection, decoder_layer_selection, decoder_model, decoder_freeze_model,
          decoder_backend, encoder_freeze,
          encoder_model,
          enable_branch1, enable_branch2, enable_branch3, head_mode,
          loss_onset_weight, loss_frame_weight, loss_velocity_weight,
          distributed, rank, local_rank, world_size, validation_include_egmd):
    if distributed:
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    if is_main_process(rank):
        print_config(ex.current_run)
        print(
            f'Loss weights (onset:frame:velocity) = '
            f'{loss_onset_weight}:{loss_frame_weight}:{loss_velocity_weight}'
        )
    
    loss_weights = (loss_onset_weight, loss_frame_weight, loss_velocity_weight)

    want_resume = (resume_from is not None) or (resume_iteration is not None)
    ckpt_dir = resolve_checkpoint_dir(resume_from, logdir) if want_resume else None
    if want_resume and resume_iteration is None:
        resume_iteration = 'auto'
    if want_resume:
        resume_iteration = resolve_resume_iteration(resume_iteration, ckpt_dir)
        if is_main_process(rank):
            print(f'Resuming from step {resume_iteration} using checkpoints in {ckpt_dir}')
            if os.path.abspath(ckpt_dir) != os.path.abspath(logdir):
                print(f'Writing continued run into new logdir {logdir}')

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir) if is_main_process(rank) else None

    train_groups, validation_groups = ['train'], ['validation']

    # Determine the actual model size used for data set initialization based on encoder selection
    #MusicFM, 95M, and 330M are treated equally, controlled by model selection parameters
    selected_model = encoder_model if encoder_model in ['MERT_95m', 'MERT_330m', 'MERT2', 'MusicFM', 'MuQ'] else None

    loader_kwargs = dict(num_workers=4, pin_memory=True)
    train_sampler = None
    if not slakh_dataset_path or not os.path.isdir(slakh_dataset_path):
        if slakh_samples_per_batch > 0 and is_main_process(rank):
            print(
                f'Warning: Slakh dataset not found at {slakh_dataset_path}; '
                f'moving {slakh_samples_per_batch} slakh slots into the matched pool.'
            )
        matched_samples_per_batch += slakh_samples_per_batch
        slakh_samples_per_batch = 0
        slakh_dataset_path = None

    if train_on == 'Groove':
        dataset = Groove(path=dataset_path, groups=train_groups, sequence_length=sequence_length)
        validation_dataset = Groove(path=dataset_path, groups=validation_groups, sequence_length=validation_length)
        groove_loader_kwargs = dict(loader_kwargs)
        if distributed:
            train_sampler = DistributedSampler(dataset, shuffle=True)
            groove_loader_kwargs['shuffle'] = False
            groove_loader_kwargs['sampler'] = train_sampler
        loader = DataLoader(dataset, batch_size, drop_last=True, **groove_loader_kwargs)
    elif train_on == 'Egmd':
        dataset = Groove(path=egmd_dataset_path, groups=train_groups, sequence_length=sequence_length)
        validation_dataset = Groove(path=egmd_dataset_path, groups=validation_groups, sequence_length=validation_length)
        egmd_loader_kwargs = dict(loader_kwargs)
        if distributed:
            train_sampler = DistributedSampler(dataset, shuffle=True)
            egmd_loader_kwargs['shuffle'] = False
            egmd_loader_kwargs['sampler'] = train_sampler
        loader = DataLoader(dataset, batch_size, drop_last=True, **egmd_loader_kwargs)
    elif train_on == 'GrooveEgmd':
        dataset = ConcatDataset(
            [
                Groove(path=dataset_path, groups=train_groups, sequence_length=sequence_length),
                Groove(path=egmd_dataset_path, groups=train_groups, sequence_length=sequence_length),
            ]
        )
        validation_dataset = ConcatDataset(
            [
                Groove(path=dataset_path, groups=validation_groups, sequence_length=validation_length),
                Groove(path=egmd_dataset_path, groups=validation_groups, sequence_length=validation_length),
            ]
        )
        concat_loader_kwargs = dict(loader_kwargs)
        if distributed:
            train_sampler = DistributedSampler(dataset, shuffle=True)
            concat_loader_kwargs['shuffle'] = False
            concat_loader_kwargs['sampler'] = train_sampler
        loader = DataLoader(dataset, batch_size, drop_last=True, **concat_loader_kwargs)
    elif train_on == 'GroupedGrooveEgmdSlakh':
        dataset = GroupedGrooveEgmdSlakhDataset(
            groove_path=dataset_path,
            egmd_path=egmd_dataset_path,
            slakh_path=slakh_dataset_path,
            groups=train_groups,
            sequence_length=sequence_length,
            matched_samples_per_batch=matched_samples_per_batch,
            unmatched_samples_per_batch=unmatched_samples_per_batch,
            slakh_samples_per_batch=slakh_samples_per_batch,
            total_batch_size=batch_size,
        )
        # Train-time val always includes Groove validation. E-GMD val default on
        # (validation_include_egmd=True).
        validation_datasets = [
            Groove(path=dataset_path, groups=validation_groups, sequence_length=validation_length),
        ]
        if validation_include_egmd:
            # Slow path (~4945 clips): only for deliberate protocol ablations, not default.
            validation_datasets.append(
                Groove(path=egmd_dataset_path, groups=validation_groups, sequence_length=validation_length)
            )
        if slakh_dataset_path:
            validation_datasets.append(
                Groove(path=slakh_dataset_path, groups=validation_groups, sequence_length=validation_length)
            )
        validation_dataset = ConcatDataset(validation_datasets)
        # Why: each grouped dataset item already returns a ready-to-train mini-batch;
        # disabling DataLoader auto-batching preserves the synchronized group layout.
        # Why: the grouped dataset keeps internal pool cursors, so grouped mode must
        # stay single-worker to avoid each worker advancing its own copied state.
        grouped_loader_kwargs = dict(loader_kwargs)
        grouped_loader_kwargs['num_workers'] = 0
        train_sampler = None
        if distributed:
            train_sampler = DistributedSampler(dataset, shuffle=True)
            grouped_loader_kwargs['shuffle'] = False
            grouped_loader_kwargs['sampler'] = train_sampler
        loader = DataLoader(dataset, batch_size=None, **grouped_loader_kwargs)
    else:
        raise ValueError(f'Unsupported train_on mode: {train_on}')

    validation_splits = [('all', validation_dataset)]
    if train_on == 'GrooveEgmd':
        validation_splits = [
            (
                'groove',
                Groove(path=dataset_path, groups=validation_groups, sequence_length=validation_length),
            ),
            (
                'egmd',
                Groove(path=egmd_dataset_path, groups=validation_groups, sequence_length=validation_length),
            ),
            ('combined', validation_dataset),
        ]

    def build_model():
        return OnsetsAndFrames(
            MAX_MIDI - MIN_MIDI + 1, model_complexity,
            encoder_model=encoder_model,
            encoder_layer_selection=encoder_layer_selection,
            # Why: decoder must use its own layer-selection config so encoder/decoder ablations stay independent.
            decoder_layer_selection=decoder_layer_selection,
            decoder_model=decoder_model,
            decoder_freeze_model=decoder_freeze_model,
            decoder_backend=decoder_backend,
            encoder_freeze=encoder_freeze,
            device=device,
            enable_branch1=enable_branch1,
            enable_branch2=enable_branch2,
            enable_branch3=enable_branch3,
            head_mode=head_mode,
        ).to(device)

    def wrap_ddp(module):
        if not distributed:
            return module
        return DDP(
            module,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    current_train_meta = build_train_meta({
        'encoder_model': encoder_model,
        'decoder_model': decoder_model,
        'decoder_backend': decoder_backend,
        'encoder_layer_selection': encoder_layer_selection,
        'decoder_layer_selection': decoder_layer_selection,
        'enable_branch1': enable_branch1,
        'enable_branch2': enable_branch2,
        'enable_branch3': enable_branch3,
        'head_mode': head_mode,
        'model_complexity': model_complexity,
        'world_size': world_size,
        'batch_size': batch_size,
        'matched_samples_per_batch': matched_samples_per_batch,
        'unmatched_samples_per_batch': unmatched_samples_per_batch,
        'slakh_samples_per_batch': slakh_samples_per_batch,
        'learning_rate': learning_rate,
        'learning_rate_decay_steps': learning_rate_decay_steps,
        'learning_rate_decay_rate': learning_rate_decay_rate,
        'encoder_freeze': encoder_freeze,
        'decoder_freeze_model': decoder_freeze_model,
        'clip_gradient_norm': clip_gradient_norm,
        'loss_onset_weight': loss_onset_weight,
        'loss_frame_weight': loss_frame_weight,
        'loss_velocity_weight': loss_velocity_weight,
    })

    restore_scheduler_state = None
    if not want_resume:
        model = wrap_ddp(build_model())
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
        best_note_f1 = 0.0  # Track the main checkpoint metric: best note F1
        best_note_f1_step = 0  # Record which training step produced the best note F1
        best_note_f1_frame_f1 = 0.0  # Keep the companion frame F1 for analysis, but not for checkpoint selection
    else:
        # Why: latest overwrite checkpoint is the source of truth; stepped filenames are legacy only.
        model_path = find_model_weights_path(ckpt_dir, resume_iteration)
        model = build_model()
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model = wrap_ddp(model)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)

        saved_meta = load_checkpoint_meta(ckpt_dir)
        saved_train_meta = saved_meta.get('train_meta', {})
        restore_plan = decide_resume_restore(saved_train_meta, current_train_meta)
        if is_main_process(rank):
            for reason in restore_plan['reasons']:
                print(f'Resume policy: {reason}')

        opt_path = os.path.join(ckpt_dir, OPTIMIZER_LATEST)
        if restore_plan['restore_optimizer'] and os.path.exists(opt_path):
            try:
                optimizer.load_state_dict(torch.load(opt_path, map_location='cpu'))
            except (ValueError, RuntimeError) as exc:
                if is_main_process(rank):
                    print(f'Warning: failed to load optimizer state ({exc}); rebuilding Adam.')
                optimizer = torch.optim.Adam(model.parameters(), learning_rate)
                restore_plan['restore_scheduler'] = False
        elif is_main_process(rank) and restore_plan['restore_optimizer']:
            print(f'Warning: {opt_path} missing; rebuilding Adam.')
            restore_plan['restore_scheduler'] = False

        sched_path = os.path.join(ckpt_dir, SCHEDULER_LATEST)
        if restore_plan['restore_scheduler'] and os.path.exists(sched_path):
            restore_scheduler_state = torch.load(sched_path, map_location='cpu')
        elif restore_plan['restore_scheduler'] and is_main_process(rank):
            print(f'Warning: {sched_path} missing; rebuilding LR schedule from config.')
            restore_plan['restore_scheduler'] = False

        if 'best_note_f1' in saved_meta:
            best_note_f1 = float(saved_meta.get('best_note_f1', 0.0))
            best_note_f1_frame_f1 = float(saved_meta.get('best_note_f1_frame_f1', 0.0))
            best_note_f1_step = int(saved_meta.get('best_note_f1_step', 0))
        else:
            best_note_f1, best_note_f1_frame_f1, best_note_f1_step = load_best_metrics_from_txt(ckpt_dir)
    
    # The MERT processor has been integrated into the data set and does not need to be managed separately

    def unwrap_model(module):
        return module.module if isinstance(module, DDP) else module

    if is_main_process(rank):
        summary(unwrap_model(model))
    # Why: last_epoch=0 makes StepLR think it is resuming and requires initial_lr on the optimizer.
    # When scheduler state is restored below, last_epoch is only a placeholder.
    # When rebuilding after an LR change, last_epoch aligns StepLR with the resumed step.
    if restore_scheduler_state is None:
        scheduler_last_epoch = (resume_iteration - 1) if resume_iteration > 0 else -1
    else:
        scheduler_last_epoch = -1
    scheduler = StepLR(
        optimizer,
        step_size=learning_rate_decay_steps,
        gamma=learning_rate_decay_rate,
        last_epoch=scheduler_last_epoch,
    )
    if restore_scheduler_state is not None:
        try:
            scheduler.load_state_dict(restore_scheduler_state)
        except (ValueError, RuntimeError, KeyError) as exc:
            if is_main_process(rank):
                print(f'Warning: failed to load scheduler state ({exc}); using rebuilt schedule.')
            scheduler = StepLR(
                optimizer,
                step_size=learning_rate_decay_steps,
                gamma=learning_rate_decay_rate,
                last_epoch=(resume_iteration - 1) if resume_iteration > 0 else -1,
            )
    loop = tqdm(range(resume_iteration + 1, iterations + 1), disable=not is_main_process(rank))
    validation_log_path = os.path.join(logdir, 'validation.log') if is_main_process(rank) else None

    def log_validation_message(message):
        # Why: plain print() shares stdout with tqdm's \r progress bar and gets overwritten.
        loop.write(message)
        if validation_log_path is not None:
            with open(validation_log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(message + '\n')

    # Initialize layer weight record file path
    layer_weights_file = os.path.join(logdir, 'mert_layer_weights.json')
    
    def save_layer_weights(model, iteration, is_best=False):
        """Save MERT layer weights to file"""
        # Get the feature extractor of the model
        feature_extractor = None
        model_for_weights = unwrap_model(model)
        if hasattr(model_for_weights, 'feature_extractor') and model_for_weights.feature_extractor is not None:
            feature_extractor = model_for_weights.feature_extractor
        
        if feature_extractor is None:
            print("Warning: Unable to find feature extractor, skip layer weight saving")
            return
        
        # Get layer weight
        layer_weights = feature_extractor.get_layer_weights()
        if layer_weights is None:
            # Get the specific layer information used
            if hasattr(feature_extractor, 'selected_layers') and feature_extractor.selected_layers:
                if len(feature_extractor.selected_layers) == 1:
                    print(f"Warning: Use single layer features (section{feature_extractor.selected_layers[0]}layer), no layer weights can be saved")
                else:
                    layers_str = ','.join(map(str, feature_extractor.selected_layers))
                    print(f"Warning: Using multi-layer features (section{layers_str}layer), there are layer weights that can be saved")
            else:
                print("Warning: Unable to obtain layer selection information, skip layer weight saving")
            return
        
        # Prepare weight data
        # Record the selected front-end name
        weights_model_name = encoder_model
        weights_data = {
            'model_size': weights_model_name,
            'num_layers': feature_extractor.num_layers,
            'iteration': iteration,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'weights': layer_weights.detach().cpu().numpy().tolist()
        }
        
        # Read existing data or create new files
        if os.path.exists(layer_weights_file):
            with open(layer_weights_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            if weights_model_name == 'MusicFM':
                layer_desc = f'MusicFM 25HzModels share{feature_extractor.num_layers}layer(12layerConformer + 1layer interface)'
            else:
                layer_desc = f'MERT v1 {weights_model_name}Models share{feature_extractor.num_layers}layer (including1layerembedding + {feature_extractor.num_layers-1}layerTransformer)'
            data = {
                'model_info': {
                    'model_size': weights_model_name,
                    'num_layers': feature_extractor.num_layers,
                    'layer_description': layer_desc
                },
                'best_model_weights': None,
                'latest_model_weights': None
            }
        
        # Update weight data
        if is_best:
            data['best_model_weights'] = weights_data
            print(f"The best model has been savedMERTlayer weight (iteration {iteration})")
        else:
            data['latest_model_weights'] = weights_data
            print(f"Latest model savedMERTlayer weight (iteration {iteration})")
        
        # Save to file
        with open(layer_weights_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def log_layer_weights_to_tensorboard(model, writer, iteration):
        """Record MERT layer weights to TensorBoard (GPU optimized version)"""
        # Get the feature extractor of the model
        feature_extractor = None
        model_for_weights = unwrap_model(model)
        if hasattr(model_for_weights, 'feature_extractor') and model_for_weights.feature_extractor is not None:
            feature_extractor = model_for_weights.feature_extractor
        
        if feature_extractor is None:
            return
        
        # Get layer weight
        layer_weights = feature_extractor.get_layer_weights()
        if layer_weights is None:
            return
        
        # GPU optimization: calculate statistics on the GPU and reduce CPU-GPU data transfer
        with torch.no_grad():
            # Compute statistics on GPU
            max_weight = torch.max(layer_weights)
            min_weight = torch.min(layer_weights)
            mean_weight = torch.mean(layer_weights)
            std_weight = torch.std(layer_weights)
            
            # Only perform one CPU conversion to obtain all required data
            weights_np = layer_weights.detach().cpu().numpy()
            max_weight_cpu = max_weight.item()
            min_weight_cpu = min_weight.item()
            mean_weight_cpu = mean_weight.item()
            std_weight_cpu = std_weight.item()
        
        # Record the weight of each layer
        for layer_idx, weight in enumerate(weights_np):
            writer.add_scalar(f'mert_layer_weights/layer_{layer_idx}', weight, global_step=iteration)
        
        # Record statistical information of weight distribution (results calculated using GPU)
        writer.add_scalar('mert_layer_weights/max_weight', max_weight_cpu, global_step=iteration)
        writer.add_scalar('mert_layer_weights/min_weight', min_weight_cpu, global_step=iteration)
        writer.add_scalar('mert_layer_weights/mean_weight', mean_weight_cpu, global_step=iteration)
        writer.add_scalar('mert_layer_weights/std_weight', std_weight_cpu, global_step=iteration)
        
        # Record weight histogram (using original GPU tensor)
        writer.add_histogram('mert_layer_weights/distribution', layer_weights, global_step=iteration)
    
    for i, batch in zip(loop, cycle(loader)):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(i)
        try:
            predictions, losses = run_transcription_batch(model, batch, loss_weights=loss_weights)
            loss = losses['loss/total']
        except Exception as e:
            print(f"Error during training at iteration {i}: {e}")
            raise e
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_gradient_norm:
            # Why: clip gradients before the optimizer step; otherwise clip_gradient_norm has no effect.
            clip_grad_norm_(model.parameters(), clip_gradient_norm)

        optimizer.step()
        scheduler.step()

        for key, value in {'loss': loss, **losses}.items():
            if writer is not None:
                writer.add_scalar(key, value.item(), global_step=i)

        # Record layer weights to TensorBoard every 500 iterations (reduce frequency to reduce GPU-CPU transfer)
        if writer is not None and i % 500 == 0:
            log_layer_weights_to_tensorboard(model, writer, i)

        if i % validation_interval == 0 and is_main_process(rank):
            model.eval()
            with torch.no_grad():
                total_loss = loss.item()
                primary_frame_f1 = 0.0
                primary_note_f1 = 0.0
                primary_onset_f1 = 0.0

                for split_name, split_dataset in validation_splits:
                    split_metrics = evaluate(split_dataset, unwrap_model(model))
                    frame_f1 = np.mean(split_metrics.get('metric/frame/f1', [0]))
                    note_f1 = np.mean(split_metrics.get('metric/note/f1', [0]))
                    onset_f1 = np.mean(split_metrics.get('metric/onset/f1', [0]))
                    note_off_f1 = np.mean(split_metrics.get('metric/note-with-offsets/f1', [0]))

                    label = split_name if split_name != 'all' else 'val'
                    log_validation_message(
                        f"Step {i:6d} [{label}] | Loss: {total_loss:.4f} | "
                        f"Frame F1: {frame_f1:.4f} | Note F1: {note_f1:.4f} | "
                        f"Onset F1: {onset_f1:.4f} | Note+off F1: {note_off_f1:.4f}"
                    )

                    tb_prefix = f'validation/{label}'
                    for key, value in split_metrics.items():
                        writer.add_scalar(
                            f'{tb_prefix}/{key.replace(" ", "_")}',
                            np.mean(value),
                            global_step=i,
                        )

                    if split_name in ('all', 'combined'):
                        primary_frame_f1 = frame_f1
                        primary_note_f1 = note_f1
                        primary_onset_f1 = onset_f1

                frame_f1 = primary_frame_f1
                note_f1 = primary_note_f1
                onset_f1 = primary_onset_f1

                if note_f1 > best_note_f1:
                    best_note_f1 = note_f1
                    best_note_f1_frame_f1 = frame_f1
                    best_note_f1_step = i
                    log_validation_message(
                        f"New best Note F1: {best_note_f1:.4f} (Frame F1: {frame_f1:.4f}) - Saving best model..."
                    )
                    save_training_checkpoint(
                        logdir,
                        unwrap_model(model),
                        optimizer,
                        scheduler,
                        i,
                        best_note_f1,
                        best_note_f1_frame_f1,
                        best_note_f1_step,
                        current_train_meta,
                        is_best=True,
                    )
                    #Save the layer weights of the optimal model
                    save_layer_weights(model, i, is_best=True)

                # Why: note F1 is the primary selection metric; frame F1 is logged with the best-note checkpoint only.
                with open(os.path.join(logdir, 'best_note_f1.txt'), 'w') as f:
                    f.write(f"Best Note F1: {best_note_f1:.4f} | Frame F1: {best_note_f1_frame_f1:.4f} | Step: {best_note_f1_step}\n")
                
            model.train()

        # Overwrite latest checkpoint only (no per-step weight history) to bound disk use.
        if i % checkpoint_interval == 0 and is_main_process(rank):
            save_training_checkpoint(
                logdir,
                unwrap_model(model),
                optimizer,
                scheduler,
                i,
                best_note_f1,
                best_note_f1_frame_f1,
                best_note_f1_step,
                current_train_meta,
                is_best=False,
            )
            #Save the layer weights of the latest model
            save_layer_weights(model, i, is_best=False)

    cleanup_distributed(distributed)
