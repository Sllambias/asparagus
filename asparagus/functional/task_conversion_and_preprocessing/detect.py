import os
from multiprocessing import Pool, Manager
from itertools import repeat
from time import time


def detect_cases(path, extension, DWI_patterns, PET_patterns, exclusion_patterns=[], processes=2):
    all_cases = []
    # 1. Build list of all files in the directory recursively.
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(extension):
                all_cases.append(os.path.join(dirpath, filename))
    print("Found %d files in %s" % (len(all_cases), path))

    # 2. Filter all files in parallel
    time_start = time()
    manager = Manager()
    L = [manager.list(), manager.list(), manager.list(), manager.list()]
    p = Pool(processes)
    p.starmap(
        filter_files,
        zip(all_cases, repeat(L), repeat(set(DWI_patterns)), repeat(set(PET_patterns)), repeat(set(exclusion_patterns))),
    )
    p.close()
    p.join()
    time_end = time()
    print("Filtering took %d seconds" % (time_end - time_start))
    return L[0], L[1], L[2], L[3]  # regular, DWI, PET, excluded


def filter_files(file, L, DWI_patterns, PET_patterns, exclusion_patterns):
    if any(exclusion_pattern in file for exclusion_pattern in exclusion_patterns):
        L[3].append(file)
    elif any(DWI_pattern in file for DWI_pattern in DWI_patterns):
        L[1].append(file)
    elif any(PET_pattern in file for PET_pattern in PET_patterns):
        L[2].append(file)
    else:
        L[0].append(file)
