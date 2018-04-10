""" Functions for reading from and writing to disk. """
import os
import pickle
import typing

import numpy as np


def parse_matrix(matrix_file: typing.TextIO) -> np.ndarray:
    return np.asarray([[int(num) for num in line.split()] for line in matrix_file])


def parse_vector(vector_file: typing.TextIO) -> np.ndarray:
    return np.asarray([int(line) for line in vector_file])


def save_many_run(many_run_out: dict, outpath: str):
    """ Save outputs from `many_run...` to a file.

    Appends if the file already exists.

    :param many_run_out: The output of many_run.simulate_all_combinations
    :param outpath: Path of the file to save the output to
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    if os.path.isfile(outpath):
        with open(outpath, 'rb') as infile:
            existing = pickle.load(infile)
    else:
        existing = {}

    for key, by_lc in many_run_out.items():
        for lc, by_lu in by_lc.items():
            for lu, predicted_language in by_lu.items():
                existing.setdefault(key, {}) \
                    .setdefault(lc, {}) \
                    .setdefault(lu, []) \
                    .append(predicted_language)
    with open(outpath, 'wb') as outfile:
        pickle.dump(existing, outfile)
