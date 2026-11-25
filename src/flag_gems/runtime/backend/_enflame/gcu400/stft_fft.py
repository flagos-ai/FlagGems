# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FFT engine for the STFT path on Enflame GCUs.

Enflame exposes no vendor FFT library, so ``flag_gems.utils.fft_intf`` had no
enflame entry and every STFT call raised ``NotImplementedError``.  torch_gcu
does implement ``aten::fft`` on GCU tensors, so this module forwards the
already-materialised ``[batch, n_fft]`` frames to it.

``norm`` convention: the forward transform is unnormalised (torch's default
"backward" normalization) and the inverse carries no ``1/n`` either (torch's
"forward" normalization), which is what ``fft_frames`` promises.
Results match the CPU reference to ~1e-7 (float32) / ~1e-6 (complex64).

Only what the STFT operator produces is accepted: contiguous float32 frames
(real input, half spectrum out) or contiguous complex64 frames (full spectrum
out; required for the inverse).  Anything else raises instead of silently
falling back to a different precision.
"""

import torch

_ACCEPTED = (torch.float32, torch.complex64)


def fft_frames(frames, *, inverse=False):
    """Transform contiguous ``[batch, n_fft]`` frames without normalization.

    Real input returns the nonredundant half spectrum; complex input returns
    all frequencies.  Inverse transforms require complex input.
    """
    if frames.device.type != "gcu":
        raise ValueError(f"STFT FFT frames must be on gcu, got {frames.device}")
    if frames.ndim != 2 or not frames.is_contiguous():
        raise ValueError("STFT FFT requires contiguous [batch, n_fft] frames")
    if frames.is_conj() or frames.is_neg():
        raise ValueError(
            "STFT FFT frames must not have unresolved conjugate/negative bits"
        )
    if inverse and not frames.is_complex():
        raise ValueError("STFT inverse FFT requires complex frames")
    if frames.dtype not in _ACCEPTED:
        raise NotImplementedError(
            f"STFT FFT does not support {frames.dtype} on enflame"
        )
    if frames.shape[1] < 1 or frames.shape[0] < 1:
        raise ValueError("STFT FFT requires nonempty frames")

    if inverse:
        # torch's "forward" normalization puts the 1/n on the forward
        # transform, which leaves ifft unnormalized -- the convention here.
        return torch.fft.ifft(frames, dim=-1, norm="forward")
    # Default ("backward") normalization leaves the forward transform
    # unnormalized and only scales the inverse.
    if frames.is_complex():
        return torch.fft.fft(frames, dim=-1)
    return torch.fft.rfft(frames, dim=-1)


def install():
    """Route the STFT FFT engine to this module on enflame.

    ``flag_gems.utils.fft_intf.fft_frames`` only knows the vendors that ship a
    direct FFT library, so on enflame every STFT call raised
    NotImplementedError.  Patching the function object, plus the reference
    ``flag_gems.ops.stft`` already bound at import time, keeps the change in
    the backend instead of adding an enflame branch to shared code.
    """
    import sys

    from flag_gems.utils import fft_intf

    if getattr(fft_intf.fft_frames, "_enflame_fft", False):
        return
    fft_frames._enflame_fft = True
    fft_intf.fft_frames = fft_frames
    for module_name in ("flag_gems.ops.stft", "flag_gems.ops.istft"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "fft_frames"):
            module.fft_frames = fft_frames
