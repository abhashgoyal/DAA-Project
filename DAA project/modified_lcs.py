import numpy as np
import time
# Load the dataset
# result = pyreadr.read_r(rf'C:\Users\adity\Desktop\DAA\Graz.rda')

# # Print the keys of the loaded data
# print("Data keys:", result.keys())


def write_dataset_to_csv(dataset_pair, index):
    X, Y = dataset_pair
    csv_file_path_Pattern =   f'dataset{index}_Pattern.csv'
    csv_file_path_Input = f'dataset{index}_Input.csv'
    np.savetxt(csv_file_path_Pattern, X, delimiter=',')
    np.savetxt(csv_file_path_Input, Y, delimiter=',')
    print(f"Dataset pair {index} has been written to {csv_file_path_Pattern} and {csv_file_path_Input}")


def calculate_similarity(X, Y, lcs_length):
    max_length = max(len(X), len(Y))
    if max_length == 0:
        return 100  # To handle the case where both series are empty
    return (lcs_length / max_length) * 100


def is_anomaly(X, Y, lcs_length, threshold=70):
    similarity = calculate_similarity(X, Y, lcs_length)
    return similarity < threshold


def mdp_lcs(X, Y, delta, search_range):
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(max(1, i - search_range), min(n + 1, i + search_range + 1)):
            # Using Euclidean distance for multivariate data 
            """
                example of how it works
                consider arrays [1,2,3] [2,3,4]
                1. difference array = [2-1,3-2,4-3]=[1,1,1]
                2. this is the length of the distance b/w to points in nD space
                3. calculating normal distance (uses normal distance formula we learn't in school but for nD space) sqrt(1^2 + 1^2 +  1^2) ~ 1.41
                so np.linalg.norm([1,2,3]-[2,3,4]) will give ~1.41
            """
            if np.linalg.norm(X[i - 1] - Y[j - 1]) <= delta:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

# Standard LCS Modified to work with to work with numerical time-series data.
def standard_lcs(X, Y,threshold=0.7):
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if np.all(np.abs(X[i - 1] - Y[j - 1]) <= threshold):
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def run_and_time_algorithms(X, Y, delta, search_range, anomaly_threshold):
    start_time = time.time()
    mdp_lcs_result = mdp_lcs(X, Y, delta, search_range)
    mdp_lcs_time = time.time() - start_time
    mdp_lcs_anomaly = is_anomaly(X, Y, mdp_lcs_result, anomaly_threshold)

    start_time = time.time()
    standard_lcs_result = standard_lcs(X, Y,delta)
    standard_lcs_time = time.time() - start_time
    standard_lcs_anomaly = is_anomaly(X, Y, standard_lcs_result, anomaly_threshold)

    return mdp_lcs_result, mdp_lcs_time, mdp_lcs_anomaly, standard_lcs_result, standard_lcs_time, standard_lcs_anomaly


np.random.seed(0) # For reproducibility (The Random datasets generated will always be the same as the seed is always 0)
dataset_pairs = [
    (np.random.rand(100, 3), np.random.rand(100, 3)),
    (np.random.rand(2000, 12), np.random.rand(2000, 12)),
    (np.random.rand(300, 2), np.random.rand(300, 2)),
    (np.random.rand(400, 3), np.random.rand(400, 3)),
    (np.random.rand(500, 3), np.random.rand(500, 3)),
    (np.random.rand(600, 3), np.random.rand(600, 3))
]
delta_example = 0.7 # Threshold
search_range_example = 10 
anomaly_threshold=70

# for index, dataset_pair in enumerate(dataset_pairs, start=1):
#     write_dataset_to_csv(dataset_pair, index)

results = []
for X, Y in dataset_pairs:
    mdp_lcs_res, mdp_lcs_time, mdp_lcs_anomaly, std_lcs_res, std_lcs_time, std_lcs_anomaly = run_and_time_algorithms(X, Y, delta_example, search_range_example, anomaly_threshold)
    results.append({
        "length": len(X),
        "mdp_lcs_res": mdp_lcs_res,
        "mdp_lcs_time": mdp_lcs_time,
        "mdp_lcs_anomaly": mdp_lcs_anomaly,
        "std_lcs_res": std_lcs_res,
        "std_lcs_time": std_lcs_time,
        "std_lcs_anomaly": std_lcs_anomaly
    })

# Print the results
for res in results:
    print(f"Length: {res['length']}, MDP LCS: {res['mdp_lcs_res']}, Time: {res['mdp_lcs_time']}, Anomaly: {res['mdp_lcs_anomaly']}, "
          f"Standard LCS: {res['std_lcs_res']}, Time: {res['std_lcs_time']}, Anomaly: {res['std_lcs_anomaly']}")

'''
    Result (time in seconds)
    Length: 100, MDP LCS: 81, Time: 0.017913818359375, Anomaly: False, Standard LCS: 90, Time: 0.06082868576049805, Anomaly: False
    Length: 2000, MDP LCS: 85, Time: 0.2202012538909912, Anomaly: True, Standard LCS: 1377, Time: 24.32408595085144, Anomaly: True
    Length: 300, MDP LCS: 273, Time: 0.03136587142944336, Anomaly: False, Standard LCS: 282, Time: 0.523796558380127, Anomaly: False
    Length: 400, MDP LCS: 332, Time: 0.032364845275878906, Anomaly: False, Standard LCS: 366, Time: 0.915276050567627, Anomaly: False
    Length: 500, MDP LCS: 411, Time: 0.03130626678466797, Anomaly: False, Standard LCS: 453, Time: 1.422328233718872, Anomaly: False
    Length: 600, MDP LCS: 494, Time: 0.06282424926757812, Anomaly: False, Standard LCS: 548, Time: 2.0594418048858643, Anomaly: False
'''


'''
    Observations:

    Standard LCS Result: The standard LCS algorithm consistently returns a result of 0. This is likely because the exact matching condition (np.array_equal) in the standard LCS is too strict for numerical data, especially for random floats, where the chance of exact matches is extremely low.

    Execution Time: The MDP-LCS algorithm is consistently faster than the standard LCS for these datasets. The time taken by both algorithms increases with the dataset size, but the increase is more significant for the standard LCS, likely due to its higher computational complexity for larger datasets.

    Applicability: This comparison shows that MDP-LCS is more suitable for numerical time-series data, especially when exact matches are improbable.
'''