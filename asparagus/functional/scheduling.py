import asparagus
import subprocess


def check_if_slurm_installed() -> bool:
    check_1 = subprocess.run(["which", "srun"], capture_output=True, text=True)
    check_2 = subprocess.run(["which", "squeue"], capture_output=True, text=True)
    if check_1.returncode == 0 and check_2.returncode == 0:
        return True


def check_if_torque_installed() -> bool:
    check_1 = subprocess.run(["which", "qsub"], capture_output=True, text=True)
    check_2 = subprocess.run(["which", "pbsnodes"], capture_output=True, text=True)
    if check_1.returncode == 0 and check_2.returncode == 0:
        return True


def check_if_lsf_installed() -> bool:
    check_1 = subprocess.run(["which", "bsub"], capture_output=True, text=True)
    check_2 = subprocess.run(["which", "bjobs"], capture_output=True, text=True)
    if check_1.returncode == 0 and check_2.returncode == 0:
        return True


def get_scheduler(mode: str = "auto"):
    if mode != "auto":
        return mode
    if check_if_slurm_installed():
        return "slurm"
    if check_if_torque_installed():
        return "torque"
    if check_if_lsf_installed():
        return "lsf"
    return "local"


def get_run_cmd_for_scheduler(scheduler: str, hardware_cfg: dict, asparagus_cmd, env_cmd) -> list:
    if scheduler == "slurm":
        return [
            "bash",
            f"{asparagus.__path__[0]}/scripts/submit_slurm.sh",
            f"{hardware_cfg.num_workers}",
            f"{hardware_cfg.num_devices}",
            env_cmd,
            asparagus_cmd,
        ]
    elif scheduler == "torque":
        raise NotImplementedError("TORQUE scheduler is not implemented yet.")
    elif scheduler == "lsf":
        return [
            "bsub",
            "-q",
            "p1",
            "-J",
            "EvalBox",
            "-R",
            "span[hosts=1]",
            "-n",
            f"{hardware_cfg.num_workers}",
            "-gpu",
            f"num={hardware_cfg.num_devices}",
            "-W",
            "12:59",
            "-R",
            "rusage[mem=4GB]",
            "-o",
            "out_%J.out",
            "-e",
            "out_%J.out",
        ]
    elif scheduler == "local":
        return [
            "bash",
            env_cmd,
            asparagus_cmd,
        ]
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}. Supported schedulers are: slurm, torque, lsf, local.")


if __name__ == "__main__":
    scheduler = get_scheduler()
    print("RESULT:", scheduler)
