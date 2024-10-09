from collections import Counter
import gzip
import json
import pickle
from typing import Any, Dict, Tuple, List, Union, Callable
import numpy as np
from pathlib import Path
import logging
from scipy.interpolate import interpn

from nogood_parser import Identifier, parse_representation_objects

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def increment_find_matrices(
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
    dimension_data: Dict[str, List[Tuple[int, int]]] = {}
    for find_var in find_json:
        if find_var.get("varType") != "matrix":
            continue
        name = find_var["name"]
        find_bins[name] = create_numpy_array(find_var)
        dimension_data[name] = get_dimension_bounds(find_var)

    for identifier, increment in increment_map.items():
        if identifier.name in find_bins:
            mapped_indices: Tuple[int, ...] = tuple(
                [
                    idx - dimension_data[identifier.name][i][0]
                    for i, idx in enumerate(identifier.indices)
                ]
            )
            find_bins[identifier.name][mapped_indices] += increment

    return find_bins


def map_values_to_nd_bins(matrix: np.ndarray, number_of_bins: int) -> np.ndarray:
    """
    This function will map values of an n-dimensional matrix to bins across all dimensions.

    Args:
    - matrix: The n-dimensional matrix to be processed.

    Returns:
    - An array where each entry is a list of tuples. Each tuple contains:
      (dimension, (bin_index, fraction)) to differentiate which dimension's bin is being referred to.
    """
    shape: Tuple[int, ...] = matrix.shape
    num_dims: int = len(shape)

    # Calculate the bin size for each dimension based on NUMBER_OF_BINS
    bin_sizes: List[float] = [dim_size / number_of_bins for dim_size in shape]

    # Create an empty map for each dimension based on the bin size
    map_to_bins: np.ndarray = np.empty(shape, dtype=object)
    for idx, _ in np.ndenumerate(matrix):
        map_to_bins[idx] = [[] for _ in range(num_dims)]

        # Calculate the bin indices and fractions for each dimension
        for dim in range(num_dims):
            value_in_dim: int = idx[dim]
            lower_bound: int = value_in_dim
            upper_bound: int = value_in_dim + 1

            bin_size: float = bin_sizes[dim]
            start_bin: int = int(lower_bound // bin_size)
            end_bin: int = int(upper_bound // bin_size)

            if start_bin == end_bin:
                # Fits entirely in one bin
                map_to_bins[idx][dim].append((start_bin, 1))
            else:
                # Spans across multiple bins, calculate fractions
                for bin_idx in range(start_bin, end_bin + 1):
                    bin_start: float = bin_idx * bin_size
                    bin_end: float = (bin_idx + 1) * bin_size

                    overlap_start: float = max(lower_bound, bin_start)
                    overlap_end: float = min(upper_bound, bin_end)

                    fraction: float = (overlap_end - overlap_start) / (
                        upper_bound - lower_bound
                    )
                    if fraction > 0:
                        map_to_bins[idx][dim].append((bin_idx, fraction))

    return map_to_bins


def apply_binning(
    binning_guide: np.ndarray, matrix: np.ndarray, number_of_bins: int
) -> np.ndarray:
    """
    Bins the values of a matrix according to the provided binning map.

    Args:
    - binning_map: The binning map created by map_values_to_nd_bins, where each entry contains
                   a list of tuples (dimension, (bin_index, fraction)).
    - matrix: The original matrix that needs to be binned.

    Returns:
    - A new binned matrix, with dimensions [NUMBER_OF_BINS, NUMBER_OF_BINS, ..., NUMBER_OF_BINS] (n-dimensional).
      Each entry contains the accumulated values based on the binning map.
    """
    num_dims = len(matrix.shape)
    binned_matrix = np.zeros([number_of_bins] * num_dims)

    for idx, value in np.ndenumerate(matrix):
        bin_contributions = binning_guide[idx]
        # Sanity check: Ensure that the fractions sum to 1
        assert all(
            [
                np.isclose(sum([fraction for _, fraction in contrib]), 1)
                for contrib in bin_contributions
            ]
        ), f"Fractions do not sum to 1: {bin_contributions}"

        # Now, combine the bin contributions from each dimension to distribute the value across bins
        def distribute_value(
            current_idx: Tuple[int, ...], current_fraction: float, dimension: int
        ):
            if dimension == num_dims:
                # When all dimensions are processed, add the value to the final bin
                binned_matrix[current_idx] += value * current_fraction
            else:
                # Recursively process each dimension
                for bin_idx, fraction in bin_contributions[dimension]:
                    distribute_value(
                        current_idx + (bin_idx,),
                        current_fraction * fraction,
                        dimension + 1,
                    )

        # Start the recursion to distribute the value across the bins
        distribute_value((), 1.0, 0)

    # Sanity check: Ensure that the sum of the original matrix is equal to the sum of the binned matrix
    original_sum = np.sum(matrix)
    binned_sum = np.sum(binned_matrix)
    assert np.isclose(
        original_sum, binned_sum
    ), f"Sum mismatch: Original sum {original_sum}, Binned sum {binned_sum}"

    return binned_matrix


def create_numpy_array(item: Dict[str, Any]) -> np.ndarray:
    """
    Create a numpy array for the given item.
    """
    if "dimensions" in item:
        shape = [dim["upper"] - dim["lower"] + 1 for dim in item["dimensions"]]
        return np.zeros(shape)
    else:
        raise ValueError("Scalar values are not supported.")


def get_dimension_bounds(item: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Get the lower and upper bounds of the dimensions for the given item.
    """
    if "dimensions" in item:
        bounds = [(dim["lower"], dim["upper"]) for dim in item["dimensions"]]
        return bounds
    else:
        raise ValueError("Scalar values are not supported.")


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


def parse_and_bin_instance(
    instance_folder_path: Path,
    aux_path: Path,
    find_path: Path,
    instance: str,
    seeds: List[str],
) -> None:
    logger.info(f"Starting to parse instance '{instance}'.")
    _, identifier_counts = parse_representation_objects(aux_path, find_path)
    logger.info(f"Parsing for instance '{instance}' completed")

    for seed in seeds:
        find_bins_path = instance_folder_path.joinpath(
            instance, seed, f"{instance}.find_bins"
        )

        if find_bins_path.exists():
            logger.info(f"Find bins already exist at '{find_bins_path}'. Skipping.")
            continue
        else:
            logger.info(f"Binning seed '{seed}' of instance '{instance}'.")
            with gzip.open(
                instance_folder_path.joinpath(instance, seed, f"{instance}.learnt.gz"),
                "rt",
            ) as f_learnt:
                learnt_lines: List[str] = f_learnt.readlines()
                f_learnt.close()
                # Skip the first line if it is a header
                learnt_clauses: List[List[int]] = [
                    list(map(int, line.strip().split(", ")[1].split()))
                    for line in learnt_lines[1:]
                ]

            bins: Dict[str, np.ndarray] = increment_find_matrices(
                find_path, identifier_counts, learnt_clauses
            )

            pickle.dump(bins, find_bins_path.open("wb"))

    logger.info(f"Finished processing instance '{instance}'.")
