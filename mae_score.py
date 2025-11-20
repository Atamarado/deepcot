# mae_score.py
import numpy as np
import datasets
import evaluate
from sklearn.metrics import mean_absolute_error

module_type = "metric"

_DESCRIPTION = """
MAE-based score for STSB-like datasets.
Score = 1 - (MAE / scale_max), default scale_max=5. Higher is better.
"""

_CITATION = " "

_INPUTS_DESCRIPTION = """
Args:
    predictions: list/np.ndarray/torch.Tensor/tf.Tensor of floats
    references:  list/np.ndarray/torch.Tensor/tf.Tensor of floats
    scale_max:   float, maximum of the gold-score scale (default 5.0)
Returns:
    dict: {'mae', 'mae_score', 'mae_score_percent'}
"""

def _to_numpy_1d(x) -> np.ndarray:
    # Accept lists, numpy arrays, torch/tf tensors
    # (Torch imports are optional; only used if available)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x, dtype=np.float64).reshape(-1)

class MAEScore():
    def _info(self):
        # Provide a non-None schema to satisfy older evaluate versions,
        # but keep it permissive (list of float64).
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_INPUTS_DESCRIPTION,
            features=datasets.Features({
                "predictions": datasets.Sequence(datasets.Value("float64")),
                "references":  datasets.Sequence(datasets.Value("float64")),
            }),
        )

    def compute(self, predictions, references, scale_max: float = 5.0):
        preds = _to_numpy_1d(predictions)
        refs  = _to_numpy_1d(references)

        if preds.shape != refs.shape:
            raise ValueError(f"Shape mismatch: predictions {preds.shape} vs references {refs.shape}")
        if scale_max <= 0:
            raise ValueError("scale_max must be > 0")

        mae = float(mean_absolute_error(refs, preds))
        score = max(0.0, min(1.0, 1.0 - mae / float(scale_max)))  # clamp to [0,1]

        return {
            "mae": mae,
            "mae_score": score,
            "mae_score_percent": 100.0 * score,
        }

Metric = MAEScore
