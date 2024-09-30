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
from collections import Counter
import json
import sys
import os
from typing import Any, Dict, Tuple, List, Union, Callable
import numpy as np
from pathlib import Path
import pickle
import gzip
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from parser import Identifier, NogoodParser, get_identifier_counts

NUMBER_OF_BINS = 10
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

    # Check if the sizes of all training data are the same
    assert len(aux_paths) == len(find_json) == len(instances_to_process)
    parse_representation_object_args = list(zip(aux_paths, find_json))

    # Parse one aux file from each instance to get the variable representations as the same representation is shared across all seeds.
    with ProcessPoolExecutor(max_workers=128) as executor:
        futures = {}
        for i, (aux_path, find_path) in enumerate(parse_representation_object_args):
            future = executor.submit(parse_representation_objects, aux_path, find_path)
            futures[future] = (instances_to_process[i], aux_path, find_path)

    for future in as_completed(futures):
        instance, aux_path, find_path = futures[future]
        try:
            result = future.result()
        except Exception as exc:
            logger.error(f"Job for item {(aux_path, find_path)} generated an exception: {exc}")
        else:
            _, identifier_counts = result
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

                    bins = increment_find_bins(find_path, identifier_counts, learnt_clauses)

                    pickle.dump(bins, find_bins_path.open("wb"))

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


def merge_shared_variables(
    find_bins: Dict[str, np.ndarray], shared_variables: List[Dict[str, Any]]
) -> Dict[str, np.ndarray]:
    """
    Merge shared variables in find_bins according to the shared_variables mapping.

    Parameters:
    - find_bins: Dictionary mapping variable names to their corresponding numpy arrays.
    - shared_variables: List of dictionaries containing shared variable mappings.
      Each dictionary should have an 'original_name' and a list of 'variables' that are to be merged.

    Returns:
    - A new dictionary with shared variables merged into a single numpy array.
    """
    # Create a copy to avoid modifying the original dictionary
    merged_bins = find_bins.copy()

    for shared_vars in shared_variables:
        original_name: str = shared_vars["original_name"]
        new_bin = None
        for var in shared_vars["variables"]:
            if var not in find_bins:
                continue  # Skip if variable is not in find_bins
            if new_bin is None:
                new_bin = np.copy(find_bins[var])
            else:
                new_bin += find_bins[var]
        if new_bin is not None:
            merged_bins[original_name] = new_bin

    return merged_bins


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        return super(CustomJSONEncoder, self).default(obj)


def increment_find_bins(
    find_json_path: Path,
    identifier_counts: Dict[int, Counter[Identifier]],
    learnt_clauses: List[List[int]],
) -> Dict[str, np.ndarray]:
    """
    Increment the find_bins arrays based on the learnt clauses.

    Parameters:
    - find_json_path: Path to the JSON file containing variable information.
    - identifier_counts: A dictionary mapping variable IDs to the number of times they are referenced.
    - learnt_clauses: A list of learnt clauses, each clause is a list of integers representing literals.

    Returns:
    - A dictionary mapping variable names to numpy arrays representing their binned counts.
    """
    increment_map: Counter[Identifier] = Counter()
    for clause in learnt_clauses:
        for nogood in clause:
            count = identifier_counts.get(abs(nogood))
            if count is None:
                continue

            increment_map += count

    with find_json_path.open("r") as f_find_json:
        find_json: List[Dict[str, Any]] = json.load(f_find_json)
        f_find_json.close()

    find_bins: Dict[str, np.ndarray] = {}
    dimension_info: Dict[str, List[Tuple[int, int]]] = {}
    for item in find_json:
        if item.get("varType") != "matrix":
            continue
        name = item["name"]
        find_bins[name] = create_numpy_array_bin(item)
        dimension_info[name] = get_dimension_info(item)

    for identifier, increment in increment_map.items():
        if identifier.name in find_bins:
            mapped_indices = tuple(
                map_to_bin(idx, *dimension_info[name][i])
                for i, idx in enumerate(identifier.indices)
            )
            find_bins[identifier.name][mapped_indices] += increment

    return find_bins


def parse_representation_objects(
    aux_path: Path, find_json_path: Path
) -> Tuple[Dict[int, str], Dict[int, Counter[Identifier]]]:
    """
    Parse representation objects from aux data and build a converted map.

    Parameters:
    - aux_path: Path to the auxiliary data file (compressed JSON).
    - find_json_path: Path to the find_json file containing variable information.

    Returns:
    - A tuple containing:
        - string_representation: A dictionary mapping variable IDs to their string representations.
        - identifier_counts: A dictionary mapping variable IDs to the number of times they are referenced in a single nogood.
    """
    with gzip.open(aux_path, "rt") as f_aux, find_json_path.open("r") as f_find_json:
        aux: Dict[str, Any] = json.load(f_aux)
        find_json: List[Dict[str, Any]] = json.load(f_find_json)
        f_aux.close()
        f_find_json.close()

    parser = NogoodParser(find_json)
    identifier_counts: Dict[int, Counter[Identifier]] = {}
    string_representation: Dict[int, str] = {}
    for key, obj in aux.items():
        if isinstance(obj, dict) and "representation" in obj:
            left_hand_side = obj.get("name", "")
            representation = obj.get("representation", "")

            parsed_nogood = parser.parse(parser.tokenize(left_hand_side))
            increments = get_identifier_counts(parsed_nogood)

            pos_val, neg_val = "", ""
            pos_op, neg_op = "=", "="

            if representation == "2vals":
                neg_val = obj.get("val1", "")
                pos_val = obj.get("val2", "")
            elif representation == "order":
                pos_val = neg_val = obj.get("value", "")
                pos_op, neg_op = "<=", ">"
            else:
                pos_val = neg_val = obj.get("value", "")
                pos_op, neg_op = "=", "!="

            positive = f"{parsed_nogood}{pos_op}{pos_val}"
            negative = f"{parsed_nogood}{neg_op}{neg_val}"

            try:
                int_key = int(key)
                string_representation[int_key] = positive
                string_representation[-int_key] = negative
                identifier_counts[int_key] = increments
            except ValueError:
                logger.warning(f"Invalid key '{key}' in aux data. Skipping.")
                continue

    return string_representation, identifier_counts


def create_numpy_array_bin(item: Dict[str, Any]) -> np.ndarray:
    """
    Create a numpy array bin for the given item.
    """
    if "dimensions" in item:
        shape = [NUMBER_OF_BINS] * len(item["dimensions"])
        return np.zeros(shape)
    else:
        raise ValueError("Scalar values are not supported.")


def get_dimension_info(item: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Get dimension information (lower and upper bounds) for the item.
    """
    if "dimensions" in item:
        return [(dim["lower"], dim["upper"]) for dim in item["dimensions"]]
    else:
        raise ValueError("Scalar values are not supported.")


def map_to_bin(value: int, lower: int, upper: int) -> int:
    """
    Map a value to a bin index based on the lower and upper bounds.
    """
    bin_size = (upper - lower + 1) / NUMBER_OF_BINS
    return min(int((value - lower) / bin_size), NUMBER_OF_BINS - 1)


def sum_normalize(arr):
    total = np.sum(arr)
    if total == 0:
        return np.zeros_like(arr)
    return (arr / total) * 100


def find_top_indices(
    heatmap: np.ndarray, n: int = 5, metric: Union[str, Callable] = "sum"
) -> List[Dict[str, Any]]:
    """
    Finds the top n indices along each dimension in an n-dimensional heatmap
    based on a specified metric.

    Parameters:
    - heatmap (array-like): n-dimensional array representing the heatmap.
    - n (int): Number of top indices to return.
    - metric (str or callable): Metric to use ('sum', or a callable function).

    Returns:
    - List of dictionaries containing 'dimension', 'index', and 'metric_value' keys.

    Raises:
    - ValueError: If an unsupported metric is provided.

    Notes:
    - The function computes the metric (default is sum) along each index of each dimension,
      and returns the top n indices across all dimensions based on the metric values.
    """
    if not isinstance(heatmap, np.ndarray):
        heatmap = np.array(heatmap)

    # Define the metric function
    if metric == "sum":
        metric_func = np.sum
    elif callable(metric):
        metric_func = metric
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    ndim = heatmap.ndim
    shape = heatmap.shape

    results = []

    # Iterate over each dimension
    for axis in range(ndim):
        indices_along_axis = range(shape[axis])
        metric_values = []

        # Compute the metric for each index along the axis
        for index in indices_along_axis:
            # Slice the heatmap at the given index along the current axis
            subarray = np.take(heatmap, indices=index, axis=axis)
            # Compute the metric
            value = metric_func(subarray)
            metric_values.append(value)

        metric_values = np.array(metric_values)

        # Find the top n indices based on the metric values
        top_n_indices = np.argsort(metric_values)[-n:][::-1]  # Descending order

        for idx in top_n_indices:
            value = metric_values[idx]
            results.append({"dimension": axis, "index": idx, "metric_value": value})

    # Sort all results by metric_value in descending order
    results_sorted = sorted(results, key=lambda x: x["metric_value"], reverse=True)

    # Return the top n results
    return results_sorted[:n]


if __name__ == "__main__":
    main()
