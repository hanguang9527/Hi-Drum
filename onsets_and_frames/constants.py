import os
import torch

# Project root: hi-drum-fresh/ (parent of onsets_and_frames/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def resolve_project_path(relative_path):
    """Resolve a path relative to the project root (hi-drum-fresh/)."""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(PROJECT_ROOT, relative_path)


# Groove / E-GMD audio is 24 kHz.
SAMPLE_RATE = 24000
# 25 Hz label grid (24000 / 25 = 960). Matches MuQ label_rate and MERT2.
# The 75 Hz cache remains on disk as h5_hop320 and is not overwritten.
HOP_LENGTH = SAMPLE_RATE // 25
# OnsetDuration length: keep withHOP_LENGTHconsistent
ONSET_LENGTH =  HOP_LENGTH
# OffsetDuration length: keep withHOP_LENGTHconsistent
OFFSET_LENGTH =HOP_LENGTH
# OnsetContinuous number of steps
HOPS_IN_ONSET =ONSET_LENGTH // HOP_LENGTH
# OffsetContinuous number of steps
HOPS_IN_OFFSET =OFFSET_LENGTH // HOP_LENGTH
# MIDIpitch range
MIN_MIDI = 21
MAX_MIDI = 108

# The mel spectrum parameter has been removed as currently pureMLPArchitecture handles directlyMERTfeature

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# MERTModel layer selection parameters
DEFAULT_LAYER_SELECTION = "all"  # By default, all layers are used for feature fusion
# Supported layer selection formats:
# - "all": Use all layers
# - single number: "0", "12" etc., using the specified single layer
# - comma separated: "0,6,12", "1,2,5,7" etc., using multi-layer fusion
