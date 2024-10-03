"""
Script to process instance directories, parse variable representations, increment bins based on learnt clauses,
and compute top indices for variables across multiple seeds and instances.

This script reads data from instance directories, processes variables, and computes heatmaps
to find the most significant variable indices based on learnt clauses from SAT solvers.

Usage:
    python script_name.py -i <instance_folder> -s <shared_variables>

Arguments:
    -i, --instance-folder: Path to the folder containing instance directories.
    -s, --shared-variables: Path to the shared_variables file.
"""

import argparse
import json
import sys
import os
from typing import Any, Dict, List, Union
import numpy as np
from pathlib import Path
import pickle
import gzip
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from binning import (
    parse_representation_objects,
    increment_find_bins,
    merge_shared_variables,
    find_top_indices,
    CustomJSONEncoder,
)

from constants import NUMBER_OF_BINS

np.set_printoptions(threshold=sys.maxsize)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    arg_parser = argparse.ArgumentParser(
        description="Process instance directories and compute top indices."
    )
    arg_parser.add_argument(
        "-i",
        "--instance-folder",
        type=str,
        required=True,
        help="Path to the folder containing instance directories.",
    )
    arg_parser.add_argument(
        "-s",
        "--shared-variables",
        type=str,
        required=True,
        help="Path to the shared_variables file.",
    )
    args = arg_parser.parse_args()

    instance_folder_path = Path(args.instance_folder)
    shared_variables_path = Path(args.shared_variables)

    if not instance_folder_path.is_dir():
        logger.error(
            f"Instance folder '{instance_folder_path}' does not exist or is not a directory."
        )
        sys.exit(1)

    if not shared_variables_path.exists():
        logger.error(f"Shared variables file not found at '{shared_variables_path}'")
        sys.exit(1)

    try:
        with shared_variables_path.open() as f_shared:
            shared_variables = json.load(f_shared)
    except Exception as e:
        logger.error(f"Failed to load shared variables file: {e}")
        sys.exit(1)

    instance_dirs = next(os.walk(instance_folder_path))[1]
    if len(instance_dirs) == 0:
        logger.error(f"No instance directories found in '{instance_folder_path}'")
        sys.exit(1)

    seeds = next(os.walk(instance_folder_path.joinpath(instance_dirs[0])))[1]

    # Check if all seeds of an instance contain the {instance}.find_bins file
    # If so, do not add it to the list
    instances_to_process = []
    for instance in instance_dirs:
        all_seeds_have_find_bins = True
        for seed in seeds:
            find_bins_path = instance_folder_path.joinpath(
                instance, seed, f"{instance}.find_bins"
            )
            if not find_bins_path.exists():
                all_seeds_have_find_bins = False
                break
        if not all_seeds_have_find_bins:
            instances_to_process.append(instance)
        else:
            logger.info(
                f"All seeds of instance '{instance}' already have find_bins. Skipping processing."
            )

    aux_paths: List[Path] = [
        instance_folder_path.joinpath(instance, seeds[0], f"{instance}.aux.gz")
        for instance in instances_to_process
    ]

    find_json: List[Path] = [
        instance_folder_path.joinpath(instance, seeds[0], f"{instance}.finds")
        for instance in instances_to_process
    ]

    # Check if the paths exist. They should :)
    assert all(aux_path.exists() for aux_path in aux_paths)
    assert all(find_path.exists() for find_path in find_json)
    # Check if the sizes of all training data are the same
    assert len(aux_paths) == len(find_json) == len(instances_to_process)
    parse_representation_object_args = list(zip(aux_paths, find_json))

    # Parse one aux file from each instance to get the variable representations as the same representation is shared across all seeds.
    with ProcessPoolExecutor(max_workers=128) as executor:
        futures_dict = {}
        for i, (aux_path, find_path) in enumerate(parse_representation_object_args):
            future = executor.submit(parse_representation_objects, aux_path, find_path)
            futures_dict[future] = (instances_to_process[i], aux_path, find_path)

        for future in as_completed(futures_dict):
            instance, aux_path, find_path = futures_dict[future]
            try:
                _, identifier_counts = future.result()
            except Exception as exc:
                logger.error(
                    f"Job for item {(aux_path, find_path)} generated an exception: {exc}"
                )
            else:
                logger.info(f"Finished processing instance '{instance}'")
                for seed in seeds:
                    find_bins_path = instance_folder_path.joinpath(
                        instance, seed, f"{instance}.find_bins"
                    )

                    if find_bins_path.exists():
                        continue
                    else:
                        with gzip.open(
                            instance_folder_path.joinpath(
                                instance, seed, f"{instance}.learnt.gz"
                            ),
                            "rt",
                        ) as f_learnt:
                            learnt_lines: List[str] = f_learnt.readlines()
                            f_learnt.close()
                            # Skip the first line if it is a header
                            learnt_clauses = [
                                list(map(int, line.strip().split(", ")[1].split()))
                                for line in learnt_lines[1:]
                            ]

                        bins = increment_find_bins(
                            find_path, identifier_counts, learnt_clauses
                        )

                        pickle.dump(bins, find_bins_path.open("wb"))

    return

    # [TODO]: Look at BSplines and use them to "bin" the data
    combined_bins: Dict[str, np.ndarray] = {}
    for instance in instance_dirs:
        for seed in seeds:
            find_bins_path = instance_folder_path.joinpath(
                instance, seed, f"{instance}.find_bins"
            )
            if not find_bins_path.exists():
                logger.error(
                    f"This should never happen. Find bins not found at '{find_bins_path}'"
                )
                sys.exit(1)
            with find_bins_path.open("rb") as f_find_bins:
                find_bins = pickle.load(f_find_bins)
                f_find_bins.close()

            find_bins = merge_shared_variables(find_bins, shared_variables)
            # Sum normalize the bins
            for name, arr in find_bins.items():
                find_bins[name] = sum_normalize(arr)
                if name in combined_bins:
                    combined_bins[name] += find_bins[name]
                else:
                    combined_bins[name] = np.copy(find_bins[name])

    top_indices_json: Dict[str, Union[int, List[Dict[str, Any]]]] = {
        "bin_size": NUMBER_OF_BINS
    }

    for name, arr in combined_bins.items():
        top_indices_json[name] = find_top_indices(arr, n=5)
        logger.debug(f"{name}:")
        logger.debug(f"Top indices: {top_indices_json[name]}")
        logger.debug(f"Array: {arr}")

    try:
        output_path = instance_folder_path.joinpath("top_indices.json")
        with output_path.open("w") as f_output:
            json.dump(
                top_indices_json,
                f_output,
                indent=4,
                cls=CustomJSONEncoder,
            )
    except Exception as e:
        logger.error(f"Failed to write top indices to '{output_path}': {e}")


if __name__ == "__main__":
    main()
