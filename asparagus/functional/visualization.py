import logging
import numpy as np
import torch
import wandb
from asparagus.functional.decorators import depends_on_mlflow
from math import ceil, floor
from PIL import Image
from typing import Any, Dict, List

try:
    import mlflow
except ImportError:
    logging.debug("MLFlow not found. Logging with MLFlow will not work.")

FPS = 4
DPI = 100


def finish():
    wandb.finish()


def update_config(dct):
    wandb.config.update(dct)


def save_tensor(x, name):
    torch.save(x, name)
    wandb.save(name)
    print(f"Saved {name} to wandb")


def normalize_array_to_pil(img: np.ndarray) -> Image.Image:
    """
    Normalize numpy array to 0-255 range and convert to PIL Image.
    Handles medical imaging data that may have negative values or wide ranges.

    Args:
        img: Numpy array representing image data

    Returns:
        PIL Image in grayscale mode ('L')
    """
    img = img.copy()
    # Normalize to 0-1 range
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:  # Avoid division by zero
        img = (img - img_min) / (img_max - img_min)
    # Convert to 0-255 range
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img, mode="L")


def get_logger_compatible_imgs(
    x, y, y_hat, slice_dim, n=1, desc="", titles=["input", "target", "prediction"]
) -> List[Dict[str, Any]]:
    """
    Generate images in a format compatible with MLflow and Wandb loggers.
    Returns a list of dictionaries with numpy arrays ready for logging.
    """
    batch_idx = 0  # we always just plot the first batch element

    x = x[batch_idx].squeeze().detach().cpu()
    y = y[batch_idx].squeeze().detach().cpu()
    y_hat = y_hat[batch_idx].squeeze().detach().cpu()

    # we might use mixed precision, so we cast to ensure tensor is compatible with numpy
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    y_hat = y_hat.to(torch.float32)

    images = []

    if n is None:
        index_range = torch.arange(x.shape[slice_dim])
    else:
        # we take the middle n images
        middle = x.shape[slice_dim] // 2
        index_range = torch.arange(middle - floor(n / 2), middle + ceil(n / 2))
        assert len(index_range) == n

    for i in index_range:
        index = (
            i if slice_dim == 0 else slice(None),
            i if slice_dim == 1 else slice(None),
        )

        xi, yi, yi_hat = (t[index].numpy() for t in (x, y, y_hat))

        # Add all three arrays to the result
        images.append(
            {
                "images": [
                    {"title": titles[0], "array": xi},
                    {"title": titles[1], "array": yi},
                    {"title": titles[2], "array": yi_hat},
                ],
                "slice_idx": i.item(),
                "caption": f"Slice {i.item()}: {desc}",
            }
        )

    return images


def log_images_to_logger(loggers, images, step, prefix=""):
    """
    Log images to MLflow or Wandb logger.

    Args:
        loggers: List of lightning logger instances
        images: List of image dictionaries from get_logger_compatible_imgs
        step: The current step/epoch (optional, if None, logger handles step automatically)
        prefix: Prefix for the image key
    """
    if not images:
        return

    for i, img_dict in enumerate(images):
        image0 = normalize_array_to_pil(img_dict["images"][0]["array"])
        image1 = normalize_array_to_pil(img_dict["images"][1]["array"])
        image2 = normalize_array_to_pil(img_dict["images"][2]["array"])

        title0 = img_dict["images"][0]["title"]
        title1 = img_dict["images"][1]["title"]
        title2 = img_dict["images"][2]["title"]

        for logger in loggers:
            logger_type = type(logger).__name__

            # WANDB
            if "WandbLogger" in logger_type:
                import wandb

                # Batch all images into a single log call to avoid hanging
                log_data = {}
                log_data[f"{prefix}/slice_{img_dict['slice_idx']}"] = [
                    wandb.Image(image0, caption=title0),
                    wandb.Image(image1, caption=title1),
                    wandb.Image(image2, caption=title2),
                ]

                logger.experiment.log(log_data, commit=False)

            # MLFLOW
            if "MLFlowLogger" in logger_type:
                prefix = prefix.replace("/", "_")
                slice_idx = img_dict["slice_idx"]
                _log_images_to_mlflow(
                    logger=logger,
                    prefix=prefix,
                    slice_idx=slice_idx,
                    step=step,
                    image0=image0,
                    image1=image1,
                    image2=image2,
                    title0=title0,
                    title1=title1,
                    title2=title2,
                )


@depends_on_mlflow()
def _log_images_to_mlflow(logger, prefix, slice_idx, step, image0, image1, image2, title0, title1, title2):
    logger.experiment.log_image(
        run_id=logger.run_id,
        image=mlflow.Image(image0),
        key=f"{prefix}_{title0}_slice_{slice_idx}",
        step=step,
    )
    logger.experiment.log_image(
        run_id=logger.run_id,
        image=mlflow.Image(image1),
        key=f"{prefix}_{title1}_slice_{slice_idx}",
        step=step,
    )
    logger.experiment.log_image(
        run_id=logger.run_id,
        image=mlflow.Image(image2),
        key=f"{prefix}_{title2}_slice_{slice_idx}",
        step=step,
    )


def get_logger_compatible_image_output_target(image, output, target, task_type: str = "segmentation"):
    channel_idx = np.random.randint(0, image.shape[0])

    if len(image.shape) == 4:  # 3D images.
        # We need to select a slice to visualize.
        if task_type == "segmentation" and len(target[0].nonzero()[0]) > 0:
            # Select a foreground slice if any exist.
            foreground_locations = target[0].nonzero()
            slice_to_visualize = foreground_locations[0][np.random.randint(0, len(foreground_locations[0]))]
        else:
            slice_to_visualize = np.random.randint(0, image.shape[1])

        image = image[:, slice_to_visualize]
        if len(target.shape) == 4:
            target = target[:, slice_to_visualize]
        if len(output.shape) == 4:
            output = output[:, slice_to_visualize]

    image = normalize_array_to_pil(image[channel_idx])

    if task_type == "classification":
        target = np.round(target.squeeze(0), decimals=3)
        output = np.round(output.argmax(0), decimals=3)
    elif task_type == "regression":
        target = np.round(target.squeeze(0), decimals=3)
        output = np.round(output.squeeze(0), decimals=3)
    elif task_type == "segmentation":
        target = target.squeeze(0)
        output = output.argmax(0)
    elif task_type == "self-supervised":
        target = normalize_array_to_pil(target[channel_idx])
        output = normalize_array_to_pil(output[channel_idx])
    else:
        logging.warn(
            f"Unknown task type. Found {task_type} and expected one in ['classification',\
                  'regression', 'segmentation', 'self-supervised']"
        )
    return image, output, target


def log_image_output_target_to_wandb(
    logger,
    image,
    output,
    target,
    log_key: str,
    fig_title: str,
    step,
    task_type: str = "segmentation",
):
    """
    Log a random image from the imagedict to wandb
    """
    if task_type in ["classification", "regression"]:
        fig = wandb.Image(image, mode="L", caption=f"P: {output} | GT: {target} | {fig_title}")
    elif task_type == "segmentation":
        fig = [
            wandb.Image(
                image,
                mode="L",
                masks={
                    "predictions": {
                        "mask_data": output,
                    },
                    "ground_truth": {
                        "mask_data": target,
                    },
                },
                caption=fig_title,
            ),
            wandb.Image(output, mode="L", caption="output"),
            wandb.Image(target, mode="L", caption="target"),
        ]
    elif task_type == "self-supervised":
        fig = [
            wandb.Image(image, mode="L", caption=fig_title),
            wandb.Image(output, mode="L", caption="output"),
            wandb.Image(target, mode="L", caption="target"),
        ]
    logger.experiment.log({log_key: fig})


def log_image_output_target_to_mlflow(
    logger,
    image,
    output,
    target,
    log_key: str,
    fig_title: str,
    step,
    task_type: str = "segmentation",
):
    """
    Log a random image from the imagedict to wandb
    """
    log_key = log_key.replace("/", "_")
    fig_title = fig_title.replace("/", "_")

    if task_type in ["classification", "regression"]:
        logger.experiment.log_image(
            run_id=logger.run_id,
            image=image,
            key=f"{log_key}_P:_{output}_|_GT:_{target}",
            step=step,
        )
    elif task_type == "segmentation":
        logger.experiment.log_image(
            run_id=logger.run_id,
            image=mlflow.Image(image),
            key=f"{log_key}_input",
            step=step,
        )
        logger.experiment.log_image(
            run_id=logger.run_id,
            image=mlflow.Image(output),
            key=f"{log_key}_output",
            step=step,
        )
        logger.experiment.log_image(
            run_id=logger.run_id,
            image=mlflow.Image(target),
            key=f"{log_key}_target",
            step=step,
        )
    elif task_type == "self-supervised":
        logger.experiment.log_image(
            run_id=logger.run_id,
            image=image,
            key=f"{fig_title}",
            step=step,
        )
        logger.experiment.log_image(
            run_id=logger.run_id,
            image=output,
            key="output",
            step=step,
        )
        logger.experiment.log_image(
            run_id=logger.run_id,
            image=target,
            key="target",
            step=step,
        )
