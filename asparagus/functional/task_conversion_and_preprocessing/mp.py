from multiprocessing.pool import Pool
from asparagus.functional.task_conversion_and_preprocessing import process_mri_case, process_dwi_case, process_pet_case
from itertools import repeat


def multiprocess_mri_dwi_pet_cases(
    files_standard,
    files_standard_out,
    pkls_standard_out,
    files_DWI,
    bvals_DWI,
    bvecs_DWI,
    files_DWI_out,
    pkls_DWI_out,
    files_PET,
    files_PET_out,
    pkls_PET_out,
    preprocessing_config,
    processes=12,
    chunksize=10,
):

    p = Pool(processes)
    p.starmap_async(
        process_mri_case,
        zip(
            files_standard,
            files_standard_out,
            pkls_standard_out,
            repeat(preprocessing_config),
        ),
        chunksize=chunksize,
    )
    p.close()
    p.join()

    p = Pool(processes)
    p.starmap_async(
        process_dwi_case,
        zip(
            files_DWI,
            bvals_DWI,
            bvecs_DWI,
            files_DWI_out,
            pkls_DWI_out,
            repeat(preprocessing_config),
        ),
        chunksize=chunksize,
    )
    p.close()
    p.join()

    p = Pool(processes)
    p.starmap_async(
        process_pet_case,
        zip(
            files_PET,
            files_PET_out,
            pkls_PET_out,
            repeat(preprocessing_config),
        ),
        chunksize=chunksize,
    )
    p.close()
    p.join()
