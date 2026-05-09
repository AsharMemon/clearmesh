import numpy as np

from scripts.research.check_face_token_leakage import _identity_retokenization_report


def test_identity_retokenization_does_not_refit_token_bin_centers():
    tokens = np.asarray(
        [
            [
                50,
                50,
                50,
                50,
                50,
                51,
                50,
                51,
                50,
            ]
        ],
        dtype=np.int64,
    )

    report = _identity_retokenization_report(tokens, num_bins=128, within_face_order="preserve")

    assert report["exact"] is True
    assert report["max_abs_delta"] == 0
    assert report["token_accuracy"] == 1.0
