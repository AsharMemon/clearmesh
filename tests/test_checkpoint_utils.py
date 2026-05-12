from pathlib import Path

import numpy as np

from clearmesh.utils.checkpoint import json_safe


def test_json_safe_removes_path_objects_from_checkpoint_metadata():
    value = json_safe(
        {
            "path": Path("/tmp/model.pt"),
            "shape": (1, np.int64(2)),
            "nested": [Path("dataset"), None],
        }
    )

    assert value == {
        "path": "/tmp/model.pt",
        "shape": [1, 2],
        "nested": ["dataset", None],
    }
