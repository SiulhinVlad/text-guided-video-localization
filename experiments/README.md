# Experimental ablations

These scripts contain research ablations and analysis tools that are not part of the main X-CLIP localization pipeline.

Run them from the repository root as Python modules so they can import the shared root-level modules:

```powershell
python -m experiments.actionness_eval --help
python -m experiments.changepoint_eval --help
python -m experiments.direct_grounder --help
python -m experiments.optical_flow_eval --help
python -m experiments.temporal_order_postprocess --help
python -m experiments.analyze_temporal_predictions --help
python -m experiments.combine_prediction_sets --help
python -m experiments.compare_prediction_sets --help
```

The scripts cover actionness and change-point signals, optical-flow augmentation, direct temporal grounding, temporal-order post-processing, and prediction-set analysis/comparison. They generally expect generated predictions, cached embeddings, datasets, or checkpoints that are intentionally excluded from Git.
