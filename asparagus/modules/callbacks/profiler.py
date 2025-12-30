"""
Lightning callback for collecting performance metrics using PyTorch profiler.
Automatically tracks forward, backward, optimizer, and data loading times.
"""

import torch
from lightning.pytorch import Callback
from torch.profiler import ProfilerActivity, profile
from typing import Any, Dict, Optional


class ProfilerCallback(Callback):
    """
    Callback that uses PyTorch profiler to collect timing metrics.

    This callback profiles training at regular intervals and logs metrics
    directly through Lightning's logging system, making them available
    in wandb, mlflow, and other configured loggers.

    Args:
        profile_every_n_steps: How often to profile (default: 100)\
    """

    def __init__(self, profile_every_n_steps: int = 100):
        self.profile_every_n_steps = profile_every_n_steps
        self.profiler: Optional[profile] = None
        self.step_count = 0

    def on_train_batch_start(self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int) -> None:
        """Start profiling if it's time to profile."""
        self.step_count += 1

        # Skip warmup steps and only profile at intervals
        if batch_idx % self.profile_every_n_steps == 0:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)

            self.profiler = profile(
                activities=activities,
                record_shapes=False,  # Minimize overhead
                profile_memory=False,  # Don't profile memory to reduce overhead
                with_stack=False,  # Don't record stack traces
            )
            self.profiler.__enter__()

    def on_train_batch_end(self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        """End profiling and extract metrics."""
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)

            # Extract timing metrics from profiler
            metrics = self._extract_timing_metrics(self.profiler)

            # Log metrics through Lightning
            if metrics:
                formatted_metrics = {f"train_performance/{k}": v for k, v in metrics.items()}
                pl_module.log_dict(
                    formatted_metrics,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                    batch_size=batch["image"].shape[0] if "image" in batch else None,
                )

            self.profiler = None

    def _extract_timing_metrics(self, profiler: profile) -> Dict[str, float]:
        """
        Extract relevant timing metrics from profiler output.

        Returns times in seconds for consistency with existing metrics.
        """
        metrics = {}

        # Get aggregated events
        key_averages = profiler.key_averages()

        # Look for specific operations
        operation_patterns = {
            "forward": ["forward", "training_step", "_forward_with_features"],
            "backward": ["backward", "autograd::engine", "loss.backward"],
            "optimizer": ["optimizer", "step", "Adam", "SGD", "zero_grad"],
            "data_loading": ["DataLoader", "__next__", "collate", "transform"],
        }

        for metric_name, patterns in operation_patterns.items():
            total_time = 0.0
            for event in key_averages:
                # Check if event name matches any pattern
                event_name_lower = event.key.lower()
                if any(pattern.lower() in event_name_lower for pattern in patterns):
                    # Use CUDA time if available, otherwise CPU time
                    if torch.cuda.is_available() and event.cuda_time_total > 0:
                        total_time += event.cuda_time_total
                    else:
                        total_time += event.cpu_time_total

            # Convert from microseconds to seconds
            metrics[f"{metric_name}_time"] = total_time / 1_000_000.0

        # Calculate total step time
        total_step_time = (
            sum(
                event.cuda_time_total if torch.cuda.is_available() and event.cuda_time_total > 0 else event.cpu_time_total
                for event in key_averages
            )
            / 1_000_000.0
        )

        metrics["step_time"] = total_step_time

        # Calculate throughput if we have step time
        if total_step_time > 0:
            # This is samples per second for a single batch
            # The actual batch size will be included via batch_size param in log_dict
            metrics["throughput_normalized"] = 1.0 / total_step_time

        # Add GPU memory metrics if using CUDA
        if torch.cuda.is_available():
            metrics["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            metrics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

        return metrics
