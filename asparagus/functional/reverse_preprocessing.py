import torch
import torch.nn.functional as F
from gardening_tools.functional.sanity_checks import verify_shapes_are_equal


def reverse_preprocessing(array, image_properties):
    canvas = torch.zeros((1, array.shape[1], *image_properties["original_size"]), dtype=array.dtype)
    pad_bbox = image_properties["pad_box"]
    crop_bbox = image_properties["crop_box"]

    shape = array.shape[2:]
    if len(shape) == 2:
        mode = "bilinear"
    elif len(shape) == 3:
        mode = "trilinear"

    if len(pad_bbox) > 0:
        array = unpad_array(array, pad_bbox)
        verify_shapes_are_equal(reference_shape=shape, target_shape=image_properties["shape_before_pad"])

    array = F.interpolate(array, size=image_properties["size_before_resample"], mode=mode)

    if len(crop_bbox) > 0:
        canvas = uncrop_array_onto_canvas(array, canvas, crop_bbox)
    else:
        canvas = array

    return canvas


def unpad_array(array, pad_box):
    """
    Unpads an array based on the provided padding box.
    Array must be shape (b,c,x,y) or (b,c,x,y,z).
    """
    if len(pad_box) == 6:
        return array[
            :,
            :,
            pad_box[0] : array.shape[2] - pad_box[1],
            pad_box[2] : array.shape[3] - pad_box[3],
            pad_box[4] : array.shape[4] - pad_box[5],
        ]
    elif len(pad_box) == 4:
        return array[:, :, pad_box[0] : array.shape[2] - pad_box[1], pad_box[2] : array.shape[3] - pad_box[3]]
    else:
        raise ValueError("Unsupported padding box length.")


def uncrop_array_onto_canvas(array, canvas, crop_bbox):
    """
    Uncrops an array onto a canvas based on the provided crop bounding box.
    Assumes arrays are shape (b, c, x, y) or (b, c, x, y, z).
    """
    slices = [
        slice(None),
        slice(None),
        slice(crop_bbox[0], crop_bbox[1] + 1),
        slice(crop_bbox[2], crop_bbox[3] + 1),
    ]
    if len(crop_bbox) == 6:
        slices.append(
            slice(crop_bbox[4], crop_bbox[5] + 1),
        )
    canvas[slices] = array
    return canvas
