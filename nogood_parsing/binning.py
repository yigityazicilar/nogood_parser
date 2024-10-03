from collections import Counter
import json
from math import e
from typing import Any, Dict, Tuple, List, Union, Callable
import numpy as np
from pathlib import Path
import gzip
import logging

from nogood_parser import Identifier, NogoodParser, get_identifier_counts
from constants import NUMBER_OF_BINS

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
    for item in find_json:
        if item.get("varType") != "matrix":
            continue
        name = item["name"]
        find_bins[name] = create_numpy_array(item)

    for identifier, increment in increment_map.items():
        if identifier.name in find_bins:
            find_bins[identifier.name][tuple(identifier.indices)] += increment

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
    logger.info(f"Processing aux data from '{aux_path}' and '{find_json_path}'")
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

            try:
                parsed_nogood = parser.tokenize(left_hand_side)
            except Exception as e:
                logger.error(f"Failed to parse '{left_hand_side}' with error: {e}")
                raise

            increments = get_identifier_counts(parsed_nogood)
            print(increments)

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
                # string_representation[int_key] = positive
                # string_representation[-int_key] = negative
                identifier_counts[int_key] = increments
            except ValueError:
                logger.warning(f"Invalid key '{key}' in aux data. Skipping.")
                continue

    logging.info(
        f"Finished processing aux data from '{aux_path}' and '{find_json_path}'. Should run the binning now."
    )
    return string_representation, identifier_counts


def create_numpy_array(item: Dict[str, Any]) -> np.ndarray:
    """
    Create a numpy array for the given item.
    """
    if "dimensions" in item:
        shape = [dim["upper"] - dim["lower"] for dim in item["dimensions"]]
        return np.zeros(shape)
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
