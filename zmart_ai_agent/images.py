"""Pictures for the model and the window: the saved files read back, their numbers, a PNG.

The agent never takes a picture itself. It asks the driver to acquire,
and the driver saves the image and answers with the paths of the files it
wrote. ``read_saved`` reads those files back: OME-TIFF (one file, or one file
per plane) and OME-Zarr (a folder holding a whole stack), the two formats
ZMART drivers save. ``image_statistics`` and ``as_png`` then turn the pixels
into a few numbers and a picture the vision model can see.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image
from pydantic_ai import BinaryContent

from .settings import LOOK_BIN, LOOK_MAX_SIDE

IMAGE_SUFFIXES = (".tif", ".tiff", ".zarr")


def saved_files(report: Any) -> list[str]:
    """The image files an acquire report names, in the order it names them.

    The ZMART contract says the report of ``acquire`` holds the saved file
    paths, but not under which key, so every text in the report that names an
    image file or folder that exists is taken. ``files`` comes first when the
    driver uses that key, as most do. A command log or a vendor's raw file is
    not an image and is left out.
    """
    found: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif (
            isinstance(value, str)
            and value.lower().endswith(IMAGE_SUFFIXES)
            and value not in found
            and Path(value).exists()
        ):
            found.append(value)

    if isinstance(report, dict):
        collect(report.get("files"))
    collect(report)
    return found


def read_saved(files: Iterable[str | Path]) -> np.ndarray:
    """The pixels of the files one acquisition saved, as one array.

    One file gives its own image (a plane, or a stack). Several files of the
    same size, such as a z-stack saved one plane per file, are stacked in the
    order given. Leading axes of length one are dropped, so a single plane is
    always two-dimensional. Raises ValueError for a file this cannot read.
    """
    planes = [_read_one(Path(path)) for path in files]
    if not planes:
        raise ValueError("the acquisition saved no image file")
    if len(planes) == 1:
        return _squeezed(planes[0])
    if len({plane.shape for plane in planes}) > 1:
        raise ValueError("the saved files hold images of different sizes")
    return _squeezed(np.stack(planes))


def _read_one(path: Path) -> np.ndarray:
    name = path.name.lower()
    if name.endswith((".tif", ".tiff")) and path.is_file():
        return tifffile.imread(path)
    if name.endswith(".zarr") and path.is_dir():
        return _read_ome_zarr(path)
    raise ValueError(
        f"cannot read {path.name}: the agent reads OME-TIFF files and OME-Zarr folders"
    )


def _squeezed(image: np.ndarray) -> np.ndarray:
    while image.ndim > 2 and image.shape[0] == 1:
        image = image[0]
    return image


def _read_ome_zarr(folder: Path) -> np.ndarray:
    """The full-resolution image of an OME-Zarr folder.

    An uncompressed Zarr (version 2), as drivers that use only the Python
    standard library write it, is read directly: it is a few JSON files and
    one raw file per piece (a "chunk") of the image. Anything else, such as
    compressed pieces or Zarr version 3, is handed to the zarr package when
    it is installed.
    """
    attributes = folder / ".zattrs"
    if attributes.is_file():
        multiscales = json.loads(attributes.read_text(encoding="utf-8"))["multiscales"][0]
        level = folder / multiscales["datasets"][0]["path"]
        meta = json.loads((level / ".zarray").read_text(encoding="utf-8"))
        if meta.get("compressor") is None and not meta.get("filters"):
            return _read_raw_chunks(level, meta)
    try:
        import zarr
    except ImportError:
        raise ValueError(
            f"cannot read {folder.name}: it is compressed or Zarr version 3, which needs the "
            "zarr package (pip install zarr)"
        ) from None
    group = zarr.open(str(folder), mode="r")
    path = group.attrs.get("multiscales", [{"datasets": [{"path": "0"}]}])[0]["datasets"][0]
    return np.asarray(group[path["path"]])


def _read_raw_chunks(level: Path, meta: dict) -> np.ndarray:
    """An uncompressed Zarr version 2 array, piece by piece."""
    shape, chunks = tuple(meta["shape"]), tuple(meta["chunks"])
    dtype = np.dtype(meta["dtype"])
    separator = meta.get("dimension_separator", ".")
    order = meta.get("order", "C")
    image = np.full(shape, meta.get("fill_value") or 0, dtype=dtype)
    grid = [-(-size // step) for size, step in zip(shape, chunks, strict=True)]
    for index in np.ndindex(*grid):
        piece = level / separator.join(str(i) for i in index)
        if not piece.is_file():
            continue  # a piece never written holds the fill value
        data = np.frombuffer(piece.read_bytes(), dtype=dtype).reshape(chunks, order=order)
        where = tuple(
            slice(i * step, min((i + 1) * step, size))
            for i, step, size in zip(index, chunks, shape, strict=True)
        )
        image[where] = data[tuple(slice(0, s.stop - s.start) for s in where)]
    return image


def _gray(image: np.ndarray) -> np.ndarray:
    """One 2-D plane: colour images are averaged, stacks are projected."""
    data = image.astype(np.float64)
    if data.ndim == 3:
        data = data[..., :3].mean(axis=-1) if data.shape[-1] in (3, 4) else data.max(axis=0)
    while data.ndim > 2:  # a stack of stacks, as several channels saved together
        data = data.max(axis=0)
    return data


def image_statistics(image: np.ndarray) -> dict[str, float]:
    """Numbers that help judge an image without a model: brightness, saturation, sharpness."""
    data = _gray(image)
    top = np.iinfo(image.dtype).max if image.dtype.kind in "ui" else image.max()
    saturated = image >= top
    colour = saturated.ndim == 3 and image.shape[-1] in (3, 4)
    while saturated.ndim > 2:  # a pixel counts once, whichever colour or plane is saturated
        saturated = saturated.any(axis=-1 if colour else 0)
        colour = False
    gy, gx = np.gradient(data)
    return {
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": round(float(data.mean()), 1),
        "saturated_percent": round(float(saturated.mean() * 100), 2),
        "sharpness": round(float(np.mean(gx**2 + gy**2) / max(data.mean(), 1.0)), 2),
    }


def as_png(image: np.ndarray, bin: int = LOOK_BIN, max_side: int = LOOK_MAX_SIDE) -> BinaryContent:
    """An 8-bit PNG for the vision model: binned, contrast stretched, at most ``max_side`` px.

    Each output pixel is the mean of a ``bin`` x ``bin`` block; a still larger
    image is binned further until it fits ``max_side``. Colour images stay in
    colour; a stack is shown as its brightest value along the stack (a maximum
    projection), so a z-stack shows everything in it.
    """
    colour = image.ndim == 3 and image.shape[-1] in (3, 4)
    data = image[..., :3].astype(np.float64) if colour else _gray(image)
    step = max(bin, int(np.ceil(max(data.shape[:2]) / max_side)))
    data = binned(data, step)
    lo, hi = np.percentile(data, (0.5, 99.5))
    scaled = np.clip((data - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(scaled).save(buffer, format="PNG")
    return BinaryContent(data=buffer.getvalue(), media_type="image/png")


def binned(data: np.ndarray, n: int) -> np.ndarray:
    """The image with each n x n block replaced by its mean (a ragged edge is dropped)."""
    if n <= 1:
        return data
    h, w = (data.shape[0] // n) * n, (data.shape[1] // n) * n
    data = data[:h, :w]
    shape = (h // n, n, w // n, n, *data.shape[2:])
    return data.reshape(shape).mean(axis=(1, 3))
