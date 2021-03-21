from __future__ import annotations

from comet_ml import Experiment

from train_cnn import train

experiment = Experiment(
    auto_metric_logging=True,
    auto_param_logging=True,
    auto_histogram_weight_logging=True,
    auto_histogram_gradient_logging=True,
    auto_histogram_activation_logging=True,
    project_name="cnn-circles",
    workspace="ahuhn",
)


if __name__ == "__main__":
    train()
