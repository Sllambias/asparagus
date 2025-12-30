from dataclasses import dataclass


@dataclass
class HardwareConfig:
    num_devices: int
    num_workers: int
    compile_mode: str = "default"
    accelerator: str = "auto"
