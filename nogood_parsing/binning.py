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


def resample_nd_matrix(matrix: np.ndarray, new_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Resample an n-dimensional matrix to a new shape using multilinear interpolation.
    """
    old_dims = matrix.shape
    dim_ranges = [np.arange(d) for d in old_dims]
    new_dims = [np.linspace(0, d - 1, num=n) for d, n in zip(old_dims, new_shape)]
    mesh_new = np.meshgrid(*new_dims, indexing="ij")

    # Flatten the coordinate grids
    coords_new = np.vstack([m.flatten() for m in mesh_new]).T

    matrix_new = interpn(
        points=dim_ranges,
        values=matrix,
        xi=coords_new,
        method="cubic",
        bounds_error=False,
        fill_value=0,
    )
    return matrix_new.reshape(new_shape)


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

            bins: Dict[str, np.ndarray] = increment_find_bins(
                find_path, identifier_counts, learnt_clauses
            )

            pickle.dump(bins, find_bins_path.open("wb"))

    logger.info(f"Finished processing instance '{instance}'.")
