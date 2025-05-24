from batchgenerators.utilities.file_and_folder_operations import join


def add_run_to_pretrained_derivative_list(ckpt_path, finetune_path):
    ckpt_path = ckpt_path.split("/version_")[0]
    ckpt_path = join(ckpt_path, "derived_models.log")
    with open(ckpt_path, "a+") as f:
        f.write(finetune_path + "\n")
