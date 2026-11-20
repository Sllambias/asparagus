import asparagus
import subprocess


def commands_exist(first_command, second_command) -> bool:
    check_1 = subprocess.run(["which", first_command], capture_output=True, text=True)
    check_2 = subprocess.run(["which", second_command], capture_output=True, text=True)
    return check_1.returncode == 0 and check_2.returncode == 0


def get_scheduler(mode: str = "auto"):
    if mode != "auto":
        return mode
    if commands_exist("srun", "squeue"):
        return "slurm"
    if commands_exist("qsub", "pbsnodes"):
        return "torque"
    if commands_exist("bsub", "bjobs"):
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
