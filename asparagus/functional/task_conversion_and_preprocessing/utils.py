def get_image_and_metadata_output_paths(
    files, source_dir, target_dir, file_suffix, image_suffix=".pt", metadata_suffix=".pkl"
):
    "Generates image file paths files and metadata paths with the input structure in the output directory"
    files_out = [f.replace(source_dir, target_dir).replace(file_suffix, image_suffix) for f in files]
    pkls_out = [f.replace(image_suffix, metadata_suffix) for f in files_out]
    return files_out, pkls_out


def get_bvals_and_bvecs_v1(files, file_suffix):
    """
    V1 assumes bvals and bvecs are in the same directory as the DWI files
    """
    bvals_out = [f.replace(file_suffix, ".bval") for f in files]
    bvecs_out = [f.replace(file_suffix, ".bvec") for f in files]
    return bvals_out, bvecs_out


def get_bvals_and_bvecs_v2():
    """
    V1 assumes bvals and bvecs are in another directory as the DWI files
    """
    raise NotImplementedError("This function is not implemented yet.")
