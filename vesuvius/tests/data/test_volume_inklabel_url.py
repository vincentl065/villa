"""Volume.download_inklabel must build the label URL with '/' on every platform.

It used os.path.dirname/join on the URL. On Windows os.path is ntpath, so the
request went to ``.../54keV_7.91um\\20230827161847_inklabels.png``, 404'd, and
the segment's ``inklabel`` was silently replaced by an all-zero placeholder.
The test runs the URL construction under ntpath on any OS.
"""

from __future__ import annotations

import io
import ntpath
import os

import numpy as np
from PIL import Image

import vesuvius.data.volume as volume_module
from vesuvius.data.volume import Volume

SEGMENT_URL = (
    "https://dl.ash2txt.org/other/dev/scrolls/1/segments/54keV_7.91um/20230827161847.zarr/"
)
LABEL_URL = (
    "https://dl.ash2txt.org/other/dev/scrolls/1/segments/54keV_7.91um/20230827161847_inklabels.png"
)


def _segment_volume() -> Volume:
    vol = Volume.__new__(Volume)
    vol.type = "segment"
    vol.url = SEGMENT_URL
    vol.verbose = False
    vol.anon = False
    vol.inklabel = None
    return vol


def _png_bytes() -> bytes:
    label = np.zeros((4, 6), dtype=np.uint8)
    label[1:3, 2:5] = 255
    buf = io.BytesIO()
    Image.fromarray(label).save(buf, format="PNG")
    return buf.getvalue()


def test_inklabel_url_uses_forward_slashes_under_windows_paths(monkeypatch) -> None:
    requested = []
    png = _png_bytes()

    def fake_open(url, mode="rb", **kwargs):
        requested.append(url)
        if url != LABEL_URL:
            raise FileNotFoundError(url)
        return io.BytesIO(png)

    monkeypatch.setattr(volume_module.fsspec, "open", fake_open)
    monkeypatch.setattr(os, "path", ntpath)  # what os.path is on Windows
    vol = _segment_volume()
    vol.download_inklabel()
    monkeypatch.undo()

    assert requested == [LABEL_URL]
    assert vol.inklabel.shape == (4, 6)
    assert int((vol.inklabel > 0).sum()) == 6
