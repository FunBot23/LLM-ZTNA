import pandas as pd
from pathlib import Path
from app.analytics.compare_models import (
    _save_bar_chart_with_error_bars, _save_confusion_matrices,
)

df = pd.read_csv("comparison_runs.csv")
results = {
    model: g.drop(columns=["model", "run"]).to_dict("records")
    for model, g in df.groupby("model", sort=False)
}
out = Path.cwd()
_save_bar_chart_with_error_bars(results, str(out / "comparison_error_bars.png"))
_save_confusion_matrices(results, str(out / "confusion_matrices.png"))
_save_confusion_matrices(results, str(out / "confusion_matrices.png"))