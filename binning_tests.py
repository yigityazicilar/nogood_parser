from icecream import ic
import numpy as np
import random

NUMBER_OF_BINS = 10

lower = 1
upper = 7

bins = np.linspace(lower, upper, NUMBER_OF_BINS + 1)
centers = [(bins[i - 1] + bins[i]) / 2 for i in range(1, len(bins))]

ic(centers)
value_distance_from_centers = []
for i in range(1, upper + 1):
    distance_from_centers = np.array(list(map(lambda x: abs(x - i), centers)))
    # value_distance_from_centers.append(distance_from_centers / np.min(distance_from_centers))
    index, value = np.argmin(distance_from_centers), np.min(distance_from_centers)
    for j, dist in enumerate(distance_from_centers):
        if dist == value:
            distance_from_centers[j] = 0
        else:
            distance_from_centers[j] = min(abs(index - j), distance_from_centers[j - 1] + 1 if j > 0 else np.inf)
    # if np.count_nonzero(distance_from_centers == 0) > 1:
    #     distance_from_centers = list(map(lambda x: x+1, distance_from_centers))
    value_distance_from_centers.append(distance_from_centers)
    ic(i, distance_from_centers)
    
value_distance_from_centers = np.array(value_distance_from_centers)
ic(value_distance_from_centers)

random.seed(0)
bins = [0] * NUMBER_OF_BINS
random_bin = [0] * (upper - lower + 1)
for _ in range(100000):
    value = random.randint(lower, upper)
    random_bin[value - lower] += 1
    for i, divisor in enumerate(value_distance_from_centers[value - lower]):
        if (divisor > 1):
            continue
        bins[i] += 1 / (2 ** divisor)

ic(bins)
# ic(random_bin)
ic(np.std(bins))