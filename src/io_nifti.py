import gzip
import struct
from pathlib import Path

import numpy as np


DT_FLOAT32 = 16
BITPIX_FLOAT32 = 32
NIFTI_HEADER_SIZE = 348
VOX_OFFSET = 352.0


def _build_header(shape, spacing=(1.0, 1.0, 1.0)):
    if len(shape) != 3:
        raise ValueError("Only 3D arrays are supported.")

    header = bytearray(NIFTI_HEADER_SIZE)
    struct.pack_into("<i", header, 0, NIFTI_HEADER_SIZE)

    dim = [3, shape[0], shape[1], shape[2], 1, 1, 1, 1]
    struct.pack_into("<8h", header, 40, *dim)
    struct.pack_into("<h", header, 70, DT_FLOAT32)
    struct.pack_into("<h", header, 72, BITPIX_FLOAT32)

    pixdim = [0.0, float(spacing[0]), float(spacing[1]), float(spacing[2]), 1.0, 1.0, 1.0, 1.0]
    struct.pack_into("<8f", header, 76, *pixdim)

    struct.pack_into("<f", header, 108, VOX_OFFSET)
    struct.pack_into("<f", header, 112, 1.0)

    # qform_code and sform_code
    struct.pack_into("<h", header, 252, 1)
    struct.pack_into("<h", header, 254, 1)

    # srow_x, srow_y, srow_z for identity affine with spacing
    struct.pack_into("<4f", header, 280, spacing[0], 0.0, 0.0, 0.0)
    struct.pack_into("<4f", header, 296, 0.0, spacing[1], 0.0, 0.0)
    struct.pack_into("<4f", header, 312, 0.0, 0.0, spacing[2], 0.0)

    header[344:348] = b"n+1\x00"
    return header


def write_nifti_gz(path, array, spacing=(1.0, 1.0, 1.0)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    header = _build_header(arr.shape, spacing=spacing)
    extension = b"\x00\x00\x00\x00"

    with gzip.open(path, "wb") as f:
        f.write(header)
        f.write(extension)
        f.write(arr.tobytes(order="C"))


def read_nifti_gz(path):
    path = Path(path)
    with gzip.open(path, "rb") as f:
        content = f.read()

    if len(content) < int(VOX_OFFSET):
        raise ValueError(f"Invalid NIfTI file: {path}")

    header = content[:NIFTI_HEADER_SIZE]
    sizeof_hdr = struct.unpack_from("<i", header, 0)[0]
    if sizeof_hdr != NIFTI_HEADER_SIZE:
        raise ValueError(f"Unsupported header size {sizeof_hdr} in {path}")

    dim = struct.unpack_from("<8h", header, 40)
    ndim = dim[0]
    if ndim != 3:
        raise ValueError(f"Only 3D NIfTI supported, got ndim={ndim} in {path}")

    sx, sy, sz = dim[1], dim[2], dim[3]
    datatype = struct.unpack_from("<h", header, 70)[0]
    if datatype != DT_FLOAT32:
        raise ValueError(f"Only float32 datatype supported, got {datatype} in {path}")

    vox_offset = int(struct.unpack_from("<f", header, 108)[0])
    raw = content[vox_offset:]

    expected = sx * sy * sz
    arr = np.frombuffer(raw, dtype=np.float32, count=expected)
    if arr.size != expected:
        raise ValueError(f"Data size mismatch for {path}")

    return arr.reshape((sx, sy, sz), order="C")
