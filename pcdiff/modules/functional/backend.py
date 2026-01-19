import os
import shutil

from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))

# Auto-detect GCC path (prefer gcc-11 for CUDA 12.x compatibility, fall back to system gcc)
_gcc_path = None
for gcc_name in ['gcc-12', 'gcc-11', 'gcc-10', 'gcc']:
    found = shutil.which(gcc_name)
    if found:
        _gcc_path = found
        break

_extra_cuda_cflags = []
if _gcc_path:
    _extra_cuda_cflags.append(f'--compiler-bindir={_gcc_path}')

_backend = load(name='_pvcnn_backend',
                extra_cflags=['-O3', '-std=c++17'],
                extra_cuda_cflags=_extra_cuda_cflags,
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'ball_query/ball_query.cpp',
                    'ball_query/ball_query.cu',
                    'grouping/grouping.cpp',
                    'grouping/grouping.cu',
                    'interpolate/neighbor_interpolate.cpp',
                    'interpolate/neighbor_interpolate.cu',
                    'interpolate/trilinear_devox.cpp',
                    'interpolate/trilinear_devox.cu',
                    'sampling/sampling.cpp',
                    'sampling/sampling.cu',
                    'voxelization/vox.cpp',
                    'voxelization/vox.cu',
                    'bindings.cpp',
                ]]
                )

__all__ = ['_backend']
