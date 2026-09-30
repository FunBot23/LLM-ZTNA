import pandas as pd
from app.analytics.compare_models import _save_bar_chart_with_error_bars

df = pd.read_csv("comparison_runs.csv")
results = {
    model: g.drop(columns=["model", "run"]).to_dict("records")
    for model, g in df.groupby("model", sort=False)
}
_save_bar_chart_with_error_bars(results, "comparison_error_bars.png")