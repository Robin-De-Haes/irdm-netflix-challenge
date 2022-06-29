### GENERAL HELPER FUNCTIONS ###

# import to save/load variables into/from a file
import pickle

def store_var(filename, variable):
    """
    Helper function to store a given variable in the given pickle file.
    """
    with open('variables/' + filename + '.pkl','wb') as pf:
        pickle.dump(variable, pf)


def load_var(filename):
    """
    Helper function to load a variable from the given pickle file.
    """
    with open('variables/' + filename + '.pkl','rb') as pf:
        return pickle.load(pf)

### MAIN CODE ###

### IMPORTS

# import needed libraries
import numpy as np
import math
import random # random number generation

import os # path manipulation
import re # regular expression matching

from scipy.sparse import csr_array, linalg # sparse matrices
from scipy.sparse.linalg import svds # computing (partial) SVD

import matplotlib.pylab as plt # plotting

from sklearn.model_selection import ParameterGrid # for hyperparameter tuning


### Task 1: Loading the Dataset

def parse_dataset_to_lists(directory_path="netflix dataset", filename_format="combined\_data\_\d+\.txt",
                           subset_prob=None):
    """
    Parse the Netflix dataset that is available at the given path into 5 lists:
    - a list of movie indices
    - a list of user indices
    - a list of ratings
    - a dict of movie ID mappings
    - a dict of user ID mappings
    with the two dict supporting bidirectional mapping from movie/user IDs to movie/user indices,
    i.e., if the movie dict contains an entry of the form 20134 -> 2 this means the movie with ID 20134 is represented
    by 2 in the list of movie indices. Furthermore, it will be stored as the second entry in the dict to also allow
    inverse mapping since dicts retain insert-order (so list(dict) can be used for inverse mappings).

    The list of movie indices, user indices and ratings have the same length with the i-th element of each list indicating
    that the i-th user provided the i-th rating for the i-th movie.

    :param directory_path: path to the directory with dataset files to be parsed
    :param filename_format: regex describing the file name format that dataset files have
    :param subset_prob: probability a dataset entry will be included in the parsed output (expected to be between 0 and 1),
                        used to subset the original dataset for development purposes

    :return: 3 lists consisting of a list of movie indices, user indices and ratings respectively
             and 2 dicts for respectively mapping movie IDs to movie indices and user IDs to user indices.
    """

    # collect the full paths to all the data files to be parsed
    file_paths = []
    for file in os.listdir(directory_path):
        if re.match("^{}$".format(filename_format), file):
            file_paths.append(os.path.join(directory_path, file))

    # initialize 2 dicts that have as keys an ID and as value an integer corresponding
    # to their representation in movie_idxs and user_idxs respectively
    movie_ids = {}
    user_ids = {}

    # initialize the 3 lists to be filled with the parsed data (for later usage to build a matrix)
    movie_idxs = []
    user_idxs = []
    ratings = []

    # movie_idxs and user_idxs should start with 0 and then increment for each new movie or user respectively
    movie_count = 0
    user_count = 0

    # process each data file
    for file_path in file_paths:
        print("processing file " + file_path)

        # we read and process the file line by line instead of reading the whole file into memory at once,
        # because the files can be very large
        with open(file_path) as fp:

            # keep track of the movie_id we're currently processing ratings for
            current_movie_index = None
            for index, line in enumerate(fp):
                if index % 1000000 == 0:
                    print("processing rating " + str(index))

                # remove leading and trailing whitespace from the line
                stripped_line = line.strip()

                # lines with a number and a colon indicate we are parsing a MovieID
                if stripped_line.endswith(":") and stripped_line[:-1].isdigit():
                    movie_id = int(stripped_line[:-1])

                    # get the index representation for this movie_id (or generate one if none exists yet)
                    movie_index = movie_ids.get(movie_id)
                    if movie_index is None:
                        # if we're subsetting it is possible a movie was stored for which no ratings were eventually
                        # outputted, so this entry should be removed from the dictionary
                        # (this is mostly code for development purposes, so its logic is kept separate from
                        #  the main code as much as possible)
                        if subset_prob is not None \
                                and movie_count > 0 \
                                and (len(movie_idxs) == 0 or movie_idxs[-1] != current_movie_index):
                            del movie_ids[current_movie_id]
                            movie_count -= 1

                        # new movie_id has been found so generate and store an index representation for it
                        movie_index = movie_count
                        movie_ids[movie_id] = movie_index
                        movie_count += 1

                    # do not store the movie in the movies list yet but just keep track of which movie we've just encountered,
                    # since we only want a movieID entry per user/rating and we haven't parsed a user/rating line
                    current_movie_index = movie_index
                    current_movie_id = movie_id  # only used when subsetting the original dataset

                # if we're taking a subset of the original dataset, there is only the specified
                # probability of subset_prob that this specific rating will be included
                elif subset_prob is None or random.random() < subset_prob:
                    # split the comma-separated values into an array
                    tokens = stripped_line.split(",")

                    # since we don't use the date we don't care if that is missing,
                    # but the first two tokens are required
                    if len(tokens) < 2:
                        print("Invalid number of values!")
                        continue

                    # check if the user_id and rating are valid values (i.e. integers)
                    try:
                        user_id = int(tokens[0])
                        rating = int(tokens[1])
                    except:
                        print("Invalid value!")
                        continue

                    # get the index representation for this user_id (or generate one if none exists yet)
                    user_index = user_ids.get(user_id)
                    if user_index is None:
                        # new user_id has been found so generate and store an index representation for it
                        user_index = user_count
                        user_ids[user_id] = user_index
                        user_count += 1

                    # store the parsed line
                    user_idxs.append(user_index)
                    movie_idxs.append(current_movie_index)
                    ratings.append(rating)

    # dicts in (modern) Python have insert-order storage so taking the keys of movie_ids and user_ids
    # will store them in a list where each ID is stored with its index corresponding to the ones in the idx lists
    return movie_idxs, user_idxs, ratings, movie_ids, user_ids


def parse_dataset(directory_path="netflix dataset", filename_format="combined\_data\_\d+\.txt", subset_prob=None):
    """
    Parse the Netflix dataset that is available at the given path into 2 sparse matrices, one of the form
    MOVIESxUSERS and one of the form USERSxMOVIES with both containing the rating for user-movie pairs as values.
    The dicts of movie IDs and user IDs corresponding with the matrix indices are returned as well, i.e., rating
    on position (5, 3) in MOVIESxUSERS corresponds to a rating for the 5th movie in movie IDs from the 3rd user
    in user IDs.

    :param directory_path: path to the directory with dataset files to be parsed
    :param filename_format: regex describing the file name format that dataset files have
    :param subset_prob: probability a dataset entry will be included in the parsed output (expected to be between 0 and 1),
                        used to subset the original dataset for development purposes

    :return: two Scipy sparse matrices (MOVIESxUSERS and USERSxMOVIES with RATINGS as data values),
             a dict of movie ID mappings and a dict of user ID mappings from ID to matrix indices
    """

    movie_idxs, user_idxs, ratings, movie_ids, user_ids = parse_dataset_to_lists(directory_path=directory_path,
                                                                                 filename_format=filename_format,
                                                                                 subset_prob=subset_prob)

    # create a sparse matrix with MOVIES rows and USERS columns with RATINGS as data
    mu_sparse_matrix = csr_array((ratings, (movie_idxs, user_idxs)), dtype=float)

    # create a sparse matrix with USERS rows and MOVIES columns with RATINGS as data,
    # if the same CSR format is used for both we could actually just take the transpose
    # of a copy of mu_sparse_matrix
    # (which would be faster, but was not done since the time needed for construction is negligible
    #  and this way we can more easily change the code to use another format here during construction)
    um_sparse_matrix = csr_array((ratings, (user_idxs, movie_idxs)), dtype=float)

    # return both the sparse matrices and the ID dicts,
    # so we can map back from an index in the matrix to the actual ID
    return mu_sparse_matrix, um_sparse_matrix, movie_ids, user_ids


### Task 2: DIMSUM

def compute_col_norms(A):
    """
    Compute the norm of each column of the given matrix A.
    
    :param A: the matrix to compute the column norms for

    :return: a list of norms for each column of the matrix A
    """
    
    # To compute the L2-norm of the columns in a sparse matrix:
    # 1) perform element-wise multiplication with itself
    # 2) perform column-wise sum
    # 3) take the square root of the sum

    return np.sqrt(A.multiply(A).sum(0))

def dimsum_map(r_i, col_norms, gamma):
    """
    Mapper function to be used in the DIMSUM algorithm.

    :param r_i: a tuple with (column indices, data values) for the non-zero elements of row i in the sparse matrix
    :param col_norms: precomputed column norms of the sparse matrix
    :param gamma: DIMSUM parameter that modulates the sampling behavior

    :return: list of tuples with emitted (key, value) pairs for the row i of the sparse matrix, with
             pairs being of the form ((j, k), a_ij * a_ik)
    """

    # extract the column indices and data values from r_i
    r_i_cols = r_i[0]
    r_i_data = r_i[1]
    
    # mapper function should emit key-value pairs, 
    # i.e., return a list of key-value pairs
    emit = []

    for j, a_ij in zip(r_i_cols, r_i_data):
        col_norm_j = col_norms[j]
        for k, a_ik in zip(r_i_cols, r_i_data):
            col_norm_k = col_norms[k]

            # compute the probability of emission
            prob_emit = min(1.0, gamma/(col_norm_j * col_norm_k))

            # emit with the specified probability
            if random.random() < prob_emit:
                emit.append(((j, k), a_ij * a_ik))

    return emit

def dimsum_combine(map_emissions):
    """
    Combiner function to be used in the DIMSUM algorithm.

    :param map_emissions: list of all emissions from dimsum_map

    :return: dictionary with the unique keys from the map_emissions and as values an array 
             of all emitted values for that key
    """

    # initialize the dict that will contain the combined emissions
    combined_emissions = {}
    
    # go over all dimsum_mappers
    for map_emission in map_emissions:
        # go over all the entries from a single dimsum_mapper
        for entry in map_emission:
            emission_key = entry[0]
            emission_val = entry[1]
            
            # add the entry to the combined_emission dict
            if emission_key in combined_emissions:
                combined_emissions[emission_key] += [emission_val]
            else:
                combined_emissions[emission_key] = [emission_val]

    return combined_emissions

def dimsum_reduce(emission_key, emission_values, col_norms, gamma):
    """
    Reducer function to be used in the DIMSUM algorithm.
    
    :param emission_key: key of the emission, should be of the form (i, j) with i and j specifying columns
    :param emission_values: list of values emitted by dimsum_map for this key
    :param col_norms: precomputed column norms of the sparse matrix
    :param gamma: DIMSUM parameter that modulates the sampling behavior

    :return: i, j and the expected cosine similarity between the columns i and j
    """

    # extract the indices and corresponding column norms for the given key
    i = emission_key[0]
    j = emission_key[1]
    col_norm_i = col_norms[i]
    col_norm_j = col_norms[j]
    
    if (gamma / (col_norm_i * col_norm_j)) > 1:
        # no approximation (all values emitted)
        b_ij = (1 / (col_norm_i * col_norm_j)) * np.sum(emission_values)
    else:
        # approximation of the expectation (via sampling)
        b_ij = (1 / gamma) * np.sum(emission_values)

    return i, j, b_ij


def dimsum(A, col_norms, gamma):
    """
    Execute the DIMSUM algorithm on the given matrix A to map it to a matrix of cosine similarities B.

    :param A: a skinny sparse matrix A to be mapped to a matrix of cosine similarities B
    :param col_norms: the precomputed column norms of matrix A
    :param gamma: DIMSUM parameter that modulates the sampling behavior (generally high for preserving singular values)

    :return: a matrix of cosine similarities B based on matrix A
    """
    
    # Some variants of DIMSUM seem to imply B should be initialized to a matrix of random numbers,
    # but they don't specify the distribution of these random numbers.
    # Therefore, we chose to initialize the cosine similarity matrix to all zeros.
    # While testing the algorithm, initializing the matrix to zeros instead of random numbers actually also 
    # gave better results.
    #B = np.random.rand(len(col_norms), len(col_norms))

    # initialize the cosine similarity matrix to all zeros,
    # this should be a smaller matrix so a regular numpy array can be used
    B = np.zeros((len(col_norms), len(col_norms)))

    # the sparse matrix is of CSR format so indptr can be used to iterate over the rows
    emitted = []
    for i in range(A.shape[0]):
        # slice one row from the sparse matrix
        pt = slice(A.indptr[i], A.indptr[i+1])
        # perform dimsum_map on the sliced row
        mapper_emission = dimsum_map((A.indices[pt], A.data[pt]), col_norms, gamma)
        # if something was actually emitted, add it to the list of emissions
        if len(mapper_emission) > 0:
            emitted.append(mapper_emission)

    # combine the mapper emissions that have the same key
    combined_emissions = dimsum_combine(emitted)

    # fill up the matrix B with the computed cosine similarity values
    for emission_key, emission_values in combined_emissions.items():
        i, j, b_ij = dimsum_reduce(emission_key, emission_values, col_norms, gamma)
        B[i][j] = b_ij
    
    return B

def approximate_AT_A(A, gamma):
    """
    Perform the DIMSUM algorithm on matrix A to obtain a matrix of cosine similarities B, 
    which is subsequently used to approximate A.T @ A via DBD 
    (with A.T being the transposed matrix of A and D being a diagonal matrix with the column norms).

    :param A: a skinny sparse matrix A for which A.T @ A should be approximated
    :param gamma: DIMSUM parameter that modulates the sampling behavior (generally high for preserving singular values)

    :return: an approximation of A.T @ A
    """
        
    # precompute the column norms (to be used in the dimsum algorithm and the generation of matrix D)
    col_norms = compute_col_norms(A)
    
    # generate matrix B with cosine similarities
    B = dimsum(A, col_norms, gamma)
    
    # generate diagonal matrix D with the column norms on its diagonal
    D = np.diag(col_norms)
    
    # approximate A.T @ A by D @ B @ D
    approximation = (D @ B) @ D
    
    return approximation

def compute_AT_A(A):
    """
    Compute the exact A.T @ A operation, with A.T being the transposed matrix of A.
    
    :param A: a skinny sparse matrix A for which A.T @ A should be computed

    :return: the result of the computation of A.T @ A, with A.T being the transposed matrix of A.
    """
    
    # A.T @ A should not be too large, so we can return it as a regular array
    return A.transpose().dot(A).toarray()

def compute_avg_MSE(exact, approximation):
    """
    Compute the average MSE over all entries for the given exact and approximated values of the A.T @ A operation,
    with A.T being the transposed matrix of A.
    
    :param exact: matrix containing the exact result of the A.T @ A operation, as returned by compute_AT_A
    :param approximation: matrix containing an approximation of the A.T @ A operation, as returned by approximate_AT_A

    :return: the average MSE over all entries for the given exact and approximated values of the A.T @ A operation,
             with A.T being the transposed matrix of A.
    """
    
    # To compute the MSE over all entries:
    # 1) perform element-wise difference between the exact matrix and the approximation
    # 2) square element-wise all values of the resulting difference matrix
    # 3) take the average over all squared differences

    return ((exact - approximation)**2).mean()


def analyze_mse_gamma(A,
                      num_runs = 5, 
                      gammas = np.linspace(10, 1000, 20), 
                      include_nz = False,
                      save_plots = False):
    """
    Helper function to analyze the impact of the DIMSUM parameter gamma on the performance of DIMSUM 
    by computing the MSE of the exact A.T @ A and the approximated results obtained via DIMSUM.
    
    A plot of the results will be shown and a matrix with the results will be returned as well.
    
    :param A: a skinny sparse matrix A for which the impact of gamma should be analyzed on the MSE for DIMSUM
    :param num_runs: the number of analytical runs to be executed 
                     (multiple runs can be executed to obtain means and standard deviations)
    :param gammas: the range of gammas to investigate the impact for
    :param include_nz: If True we will also compute the MSE taking only non-zero entries into account.
                       In our system zero entries will always be correct which leads to a lower MSE
                       in our analysis if the mean is computed taking these zero entries into account.
    :param save_plots: if True plots of the analysis results will be saved in a local directory.
    
    :return: matrix of shape (num_runs, len(gammas)) with the MSE results as values
             and if include_nz is True we also get a similar matrix with MSE results that only take non-zero entries 
             into account (or None if include_nz is False)
    """
        
    # exact A.T @ A is always the same (independent of the run or gamma)
    exact = compute_AT_A(A)
    
    # compute and store the MSEs for the specified number of runs and gammas
    MSEs = np.empty((num_runs, len(gammas)))
    for run in range(num_runs):
        print("run " + str(run+1))
        for gamma_index, gamma in enumerate(gammas):
            approximation = approximate_AT_A(A, gamma)

            MSEs[run, gamma_index] = compute_avg_MSE(exact, approximation)
    
    MSEs_to_plot = [MSEs]
    plot_titles = ['Impact of $\gamma$ on MSE\n' + "(averaged over " + str(num_runs) + " runs)\n"]
    file_paths = ["graphs/impact_gamma_mse.png"]
    
    # zero entries gave MSEs of 0, so we can simply recompute the non-zero MSEs by modifying the denominator
    MSEs_nz = MSEs * exact.size / np.count_nonzero(exact) if include_nz else None
    if include_nz:
        MSEs_to_plot.append(MSEs_nz)
        plot_titles.append('Impact of $\gamma$ on MSE of non-zero entries\n' + "(averaged over " + str(num_runs) + " runs)\n")
        file_paths.append("graphs/impact_gamma_mse_nz.png")
    
    # Fonts that are used in the plots
    font_small = 13
    font_medium = 14
    font_large = 15
    
    for MSE_results, plot_title, file_path in zip(MSEs_to_plot, plot_titles, file_paths):
        
        fig, axes = plt.subplots(figsize = (11, 9))

        # compute means and standard deviations per gamma (over all runs)
        mse_means = np.mean(MSE_results, 0)
        mse_sds = np.std(MSE_results, 0)

        # plot the results
        plt.plot(gammas, 
                 mse_means,
                 color = "blue",
                 zorder = 5)
        plt.fill_between(gammas, 
                         mse_means - mse_sds, 
                         mse_means + mse_sds,
                         alpha = 0.3,
                         color = "blue",
                         zorder = 0)

        plt.xticks(fontsize = font_small)
        plt.yticks(fontsize = font_small)

        plt.ylim(bottom = 0)
        plt.xlim(left = 0)

        plt.xlabel('\n$\gamma$', fontsize = font_medium)
        plt.ylabel('MSE\n', fontsize = font_medium)
        plt.title(plot_title, fontsize = font_large)

        if save_plots:
            plt.savefig(file_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    return MSEs, MSEs_nz


# (code for Task 3 is available below the code for Task 4)

### Task 4: Accuracy

def compute_prediction(Q, P_t, i, j):
    """
    Compute a prediction for the value at position (i, j) in a matrix A by multiplying the i-th row of Q with the
    j-th column of P_t, with Q and P_t being latent factor matrices of A.
    
    :param Q: Q component of a sparse matrix
    :param P_t: P_t component of a sparse matrix
    :param i: row index in A of the entry to predict
    :param j: column index in A of the entry to predict

    :return: a prediction of the value at position (i, j) in a sparse matrix based on the Q and P_t components of this matrix
    """

    # multiply the i-th row of Q with the j-th column of P_t to obtain a prediction
    return np.dot(Q[i, :], P_t[:, j])


def compute_RMSE(A, Q, P_t):
    """
    Compute the RMSE over all entries for the original matrix A with regard to the predictions obtained via
    Q and P_t.
    
    :param A: a sparse matrix A
    :param Q: Q component of sparse matrix A
    :param P_t: P_t component of sparse matrix A

    :return: the RMSE over all entries for the original matrix A with regard to the predictions obtained via
             Q and P_t.
    """
    
    # convert to COO format, since it's easier to go over all non-zero entries that way
    A_coo = A.tocoo()
    
    # keep track of the total number of non-zero entries, i.e., the number of squared errors we'll compute,
    # so we can easily average later
    SE_count = A_coo.nnz

    # keep track of the sum of squared errors
    SE_sum = 0

    # go over all non-zero entries in the original matrix
    for i, j, d in zip(A_coo.row, A_coo.col, A_coo.data):
        # compute the squared error and add it to the sum of squared errors
        SE_sum += (d - compute_prediction(Q, P_t, i, j))**2

    # compute the mean of the squared errors (SE_sum/SE_count) and then take the root of the result to obtain the RMSE    
    return np.sqrt(SE_sum / SE_count)


def train_test_relative_split(A, relative_test_size=0.20, random_seed=None):
    """
    Split the sparse matrix A in a training set and a test, with the test set containing relative_test_size 
    non-zero entries from A and the training set containing the rest of the non-zero entries.   

    :param A: sparse matrix that should be split into a training and a test set
    :param relative_test_size: relative size of non-zero entries of A that should be put in the test set,
                               the value is expected to be in [0;1]
    :param random_seed: seed used for controlling the shuffling applied to the data before splitting,
                        an int should be passed for reproducible output.

    :return: a sparse matrix representing the training set and a sparse matrix representing the test set
    """
    
    # the COO format gives easier access to the indices of non-zero entries,
    # which simplifies making a good split
    A_coo = A.tocoo()

    # get the number of non-zero entries in A
    num_nz_data = A_coo.nnz

    # get the rows, cols and data of non-zero entries
    rows = A_coo.row
    cols = A_coo.col
    data = A_coo.data
    
    # set the random seed to be used for the generation of indices to use for the test set
    random.seed(random_seed)

    # randomly select entry indices to use for the test set
    test_idxs = random.sample(range(num_nz_data), round(num_nz_data * relative_test_size))

    # the test set consists of the entries in the original matrix A specified by test_idxs
    test = csr_array((data[test_idxs], (rows[test_idxs], cols[test_idxs])), shape=A.shape)

    # the training set consists of the values in the original matrix A that are not present in the test set,
    # which can be obtained by taking the difference between the original matrix and the test matrix
    train = A - test
    
    return train, test


def parse_test_file(movie_id_dict, user_id_dict, directory_path="netflix dataset", filename="probe.txt"):
    """
    Parse the given test file to obtain a list of movie and user indices that belong to a test set.
    
    The list of movie indices and user indices have the same length with the i-th element of each list indicating
    that the i-th user and the i-th movie entry in the original sparse matrix belong tot a test set.
    
    :param movie_id_dict: a dict of movie ID mappings (from movie ID to index in the original sparse matrix)
    :param user_id_dict: a dict of user ID mappings (from user ID to index in the original sparse matrix)
    :param directory_path: path to the directory with dataset files to be parsed
    :param filename: filename of the file containing the movie and user ID combinations of a test set

    :return: a list of movie indices and a list of user indices of the same length,
             with the combined i-th entries representing an entry in the original sparse matrix that belongs to a test set.
    """

    # find the full path to the test file
    file_path = os.path.join(directory_path, filename)

    # initialize 2 lists containing the indices corresponding to entries in the original matrix
    movie_idxs = []
    user_idxs = []

    # we read and process the file line by line instead of reading the whole file into memory at once,
    # because the file could be very large
    with open(file_path) as fp:
        
        # keep track of the movie_id we're currently processing ratings for
        current_movie_index = None
        for index, line in enumerate(fp):           
            # remove leading and trailing whitespace from the line
            stripped_line = line.strip()

            # lines with a number and a colon indicate we are parsing a MovieID
            if stripped_line.endswith(":") and stripped_line[:-1].isdigit():
                movie_id = int(stripped_line[:-1])

                # get the index representation for this movie_id (or print a warning if none exists)
                movie_index = movie_id_dict.get(movie_id) 
                if movie_index is None:
                    print("Movie not present in original dataset: " + str(movie_id))
                    continue

                # do not store the movie in the movies list yet but just keep track of which movie we've just encountered,
                # since we only want a movieID entry per user and we haven't parsed a user line
                current_movie_index = movie_index
            else:
                # check if the user_id is a valid value (i.e. an integer)
                try:
                    user_id = int(stripped_line)
                except:
                    print("Invalid value!")
                    continue   

                # get the index representation for this user_id (or print a warning if none exists)
                user_index = user_id_dict.get(user_id) 
                if user_index is None:
                    print("User not present in original dataset: " + str(user_id))
                    continue

                # store the parsed line
                user_idxs.append(user_index)
                movie_idxs.append(current_movie_index)

    return movie_idxs, user_idxs


def train_test_split(A, row_idxs_test, col_idxs_test):
    """
    Split off a test set from sparse matrix A based on the given lists of movie and user indices that belong in the
    test set. The i-th element of row_idxs_test and col_idxs_test indicate a row-col pair that should be moved
    to the test set.

    :param A: sparse matrix that should be split into a training and a test set
    :param row_idxs_test: list of indices that represent the row index part of each test pair
    :param col_idxs_test: list of indices that represent the col index part of each test pair

    :return: a sparse matrix representing the training set and a sparse matrix representing the test set
    """
    
    # generate a mask that specifies which ratings in A should belong to the test set,
    # i.e., all ratings with corresponding indices in row_idxs_test and col_idxs_test
    test_mask = csr_array((np.ones(len(row_idxs_test)), (row_idxs_test, col_idxs_test)), shape=A.shape)
    
    # the test set consists of the values in the original matrix A that have a "1" in the generated mask,
    # so elementwise multiplication should remove all the other values and only keep the test set
    test = A.multiply(test_mask)
    
    # the training set consists of the values in the original matrix A that are not present in the test set,
    # which can be obtained by taking the difference between the original matrix and the test matrix
    train = A - test
    
    return train, test


### Task 3: (Stochastic) Gradient Descent with Latent Factors

def stochastic_gradient_descent(A, Q, P_t, num_epochs=100, eta=1e-5, lambda_=1e-2, shuffle=True,
                                include_RMSE=False, A_test=None):
    """
    Implementation of stochastic gradient descent.

    The idea is to minimize the error between the actual ratings and the estimates in a regularized approach. We solve
    this minimization problem by looping through all ratings in the training set, computing the predicted rating
    and afterwards computing the error between the prediction and the actual rating.
    After each rating we then modify Q and P_t accordingly:
    q_i <- q_i + eta * (e_ji * p_j - lambda * q_i)
    p_j <- p_j + eta * (e_ji * q_i - lambda * p_j)

    :param A: sparse matrix containing samples
    :param Q: U matrix obtained via SVD (note Q = U)
    :param P_t: P.T matrix obtained via SVD (note P.T = np.diag(s) @ V.T)
    :param num_epochs: number of iterations to perform over all samples
    :param eta: learning rate
    :param lambda_: regularization parameter
    :param shuffle: if True the input data will be shuffled each epoch before descending further
    :param include_RMSE: if True, the RMSE will be computed on the training data A and test data A_test (if its given)
    :param A_test: test set to compute the RMSE on for validation purposes

    :return: the iteratively refined matrices Q and P.T (i.e. the final factor estimates we obtained)
             and a list with the RMSE per epoch for both A and A_test
    """

    # convert to COO format, since it's easier to go over all non-zero entries that way
    A_coo = A.tocoo()

    # we also convert A_test to COO as it will speed up RMSE computation inside compute_RMSE
    A_test_coo = None if A_test is None else A_test.tocoo()

    # we create a copy so the original matrices don't get modified
    # (this is done for convenience but also needs more memory and is not strictly necessary)
    Q_new = Q.copy()
    P_new = P_t.copy()

    # initialize RMSEs (that we track per epoch)
    if include_RMSE:
        RMSEs = [compute_RMSE(A_coo, Q_new, P_new)]
        RMSEs_test = None if A_test is None else [compute_RMSE(A_test_coo, Q_new, P_new)]
    else:
        RMSEs = None
        RMSEs_test = None

    # We generate an array with an index for each non-zero element to easily shuffle later on.
    # Alternatively we could use: 
    # random.sample(list(zip(A_coo.row, A_coo.col, A_coo.data)), A_coo.nnz) if shuffle else zip(A_coo.row, A_coo.col, A_coo.data)
    # but this takes up more memory and made our stochastic gradient descent perform noticeably slower.
    # We could still use zip(A_coo.row, A_coo.col, A_coo.data) in case shuffle is False as we don't need the indices array
    # if we're not shuffling the data, but the memory usage of this numpy array seems non-problematic.
    # In addition, not having to make the distinction between an array or a zip depending on a shuffle boolean makes our code clearer
    # so we chose to always create it.
    indices = np.arange(A_coo.nnz)

    # repeat for the specified number of epochs
    for epoch in range(0, num_epochs):
        print("epoch: " + str(epoch+1))

        # in practice it is often recommended to perform random shuffling to avoid potential cycles
        # that might introduce bias
        if shuffle: 
            random.shuffle(indices)

        for idx in indices:
            # get the specified matrix entry (its indices and rating)
            i = A_coo.row[idx]
            j = A_coo.col[idx]
            d = A_coo.data[idx]

            # compute the prediction error
            predicted_rating = compute_prediction(Q_new, P_new, i, j)
            prediction_error = d - predicted_rating

            # update Q and P based on the gradient for this sample
            Q_new[i, :] = Q_new[i, :] + eta * (prediction_error * P_new[:, j] - lambda_ * Q_new[i, :])
            P_new[:, j] = P_new[:, j] + eta * (prediction_error * Q_new[i, :] - lambda_ * P_new[:, j])

        # we compute and store the RMSE after every epoch
        if include_RMSE:
            RMSEs.append(compute_RMSE(A_coo, Q_new, P_new))
            if RMSEs_test is not None:
                RMSEs_test.append(compute_RMSE(A_test_coo, Q_new, P_new))

    return Q_new, P_new, RMSEs, RMSEs_test


def batch_gradient_descent(A, Q, P_t, num_epochs=100, eta=0.1, lambda_=1e-2, include_RMSE=False, A_test=None):
    """
    Implementation of batch gradient descent.

    The idea is to minimize the error between the actual ratings and the estimates in a regularized approach. We solve
    this minimization problem by looping through all ratings in the training set, computing for each one the error
    between the prediction we make and the actual rating and finally summing up the gradients we can obtain this way.
    After calculating the gradient over all ratings, we modify Q and P.T accordingly:
    q_i <- q_i + eta * q_grad where q_grad = SUM(e_ji * p_j - lambda * q_i)
    p_j <- p_j + eta * p_grad where p_grad = SUM(e_ji * q_i - lambda * p_j)

    :param A: sparse matrix containing samples
    :param Q: U matrix obtained via SVD (note Q = U)
    :param P_t: P.T matrix obtained via SVD (note P = np.diag(s) @ V.T)
    :param num_epochs: number of iterations to perform over all samples
    :param eta: learning rate
    :param lambda_: regularization parameter
    :param include_RMSE: if True, the RMSE will be computed on the training data A and test data A_test (if its given)
    :param A_test: test set to compute the RMSE on for validation purposes

    :return: the iteratively refined matrices Q and P.T (i.e. the final factor estimates we obtained)
             and a list with the RMSE per epoch for both A and A_test
    """

    # convert to COO format, since it's easier to go over all non-zero entries that way
    A_coo = A.tocoo()

    # we also convert A_test to COO as it will speed up RMSE computation inside compute_RMSE
    A_test_coo = None if A_test is None else A_test.tocoo()

    # we create a copy so the original matrices don't get modified
    # (this is done for convenience but also needs more memory and is not strictly necessary)
    Q_new = Q.copy()
    P_new = P_t.copy()

    # initialize RMSEs (that we track per epoch)
    if include_RMSE:
        RMSEs = [compute_RMSE(A_coo, Q_new, P_new)]
        RMSEs_test = None if A_test is None else [compute_RMSE(A_test_coo, Q_new, P_new)]
    else:
        RMSEs = None
        RMSEs_test = None

    # repeat for the specified number of epochs
    for epoch in range(0, num_epochs):
        print("epoch: " + str(epoch+1))

        # initialize the gradients to zero values for the current epoch
        Q_gradient = np.zeros(Q_new.shape)
        P_gradient = np.zeros(P_new.shape)

        # in case of batch gradient descent, no random shuffling is needed 
        # since we always go over all samples before performing an update
        for i, j, d in zip(A_coo.row, A_coo.col, A_coo.data):
            # compute the prediction error
            predicted_rating = compute_prediction(Q_new, P_new, i, j)
            prediction_error = d - predicted_rating

            # track the gradients for this sample
            Q_gradient[i, :] += (prediction_error * P_new[:, j] - lambda_ * Q_new[i, :])
            P_gradient[:, j] += (prediction_error * Q_new[i, :] - lambda_ * P_new[:, j])

        # update Q and P.T based on the total gradient over all samples
        Q_new = Q_new + eta * Q_gradient
        P_new = P_new + eta * P_gradient

        # we compute and store the RMSE after every epoch
        if include_RMSE:
            RMSEs.append(compute_RMSE(A_coo, Q_new, P_new))
            if RMSEs_test is not None:
                RMSEs_test.append(compute_RMSE(A_test_coo, Q_new, P_new))

    return Q_new, P_new, RMSEs, RMSEs_test


def analyze_k_gradient_descent(A_train,
                               A_test,
                               gd_type='sgd',
                               ks = np.arange(10, 101, 10), 
                               num_epochs=2,
                               lambda_=1e-2,
                               eta=1e-5,
                               save_plots = False,
                               **kwargs):
    """
    Helper function to analyze the impact of the parameter k on the performance of gradient descent.
    The performance will be measured by computing the RMSE after each epoch on the training set and a validation set.
    
    :param A_train: a sparse training matrix to compute the in-sample error
    :param A_test: a sparse test matrix to compute the out-of-sample error for
    :param gd_type: the type of gradient descent to perform (expected to be either 'sgd' or 'bgd')
    :param ks: the range of k values to investigate the impact for
    :param num_epochs: the number of epochs that should be executed by gradient descent
    :param lambda_: the regularization parameter that should be used in the gradient descent objective function
    :param eta: the learning rate that should be used in gradient descent
    :param save_plots: if True plots of the analysis results will be saved in a local directory. 
    :param kwargs: any additional parameters that should be passed to the gradient descent function
    
    :return: 4 lists with the first two lists containing the RMSE results per k on the training and test set respectively
             and the last two containing respectively the final Q and P.T matrices that are obtained per k
    """

    # computing RMSE will go quicker on a COO version of the matrix in our compute_RMSE implementation
    A_train_coo = A_train.tocoo()
    A_test_coo = A_test.tocoo()

    # store the RMSEs, Qs and P.Ts for the training and test set for the specified ks
    RMSEs = []
    RMSEs_test = []
    Qs = []
    Ps = []
    for k in ks:
        print("k: " + str(k))
        
        # compute the largest k singular values and corresponding vectors of SVD on the training set
        U, s, V_t = linalg.svds(A_train, k, which='LM')

        # pre-multiply the components to obtain initial Q and P.T,
        # Q and P.T are smaller than A and can be stored as regular numpy arrays
        Q = np.array(U, dtype=np.float64)
        P_t = np.array(np.diag(s) @ V_t, dtype=np.float64)
        
        if gd_type == 'sgd':
            # perform stochastic gradient descent
            Q_new, P_new, _, _ = stochastic_gradient_descent(A_train, Q, P_t,
                                                             num_epochs=num_epochs, eta=eta,
                                                             lambda_=lambda_, include_RMSE=False,
                                                             A_test=A_test, **kwargs)
        else:
            # perform batch gradient descent
            Q_new, P_new, _, _ = batch_gradient_descent(A_train, Q, P_t,
                                                        num_epochs=num_epochs, eta=eta,
                                                        lambda_=lambda_, include_RMSE=False,
                                                        A_test=None, **kwargs)

        # store the final RMSE for this k on the training set and test set
        RMSEs.append(compute_RMSE(A_train_coo, Q_new, P_new))
        RMSEs_test.append(compute_RMSE(A_test_coo, Q_new, P_new))

        # store the final obtained Q and P.T for this k
        Qs.append(Q_new)
        Ps.append(P_new)

    # Fonts that are used in the plots
    font_small = 13
    font_medium = 14
    font_large = 15
    
    fig, axes = plt.subplots(figsize = (11, 9))

    # plot the results
    plt.plot(ks, 
             RMSEs,
             label="Training RMSE",
             color = "blue",
             zorder = 1)
    plt.plot(ks, 
             RMSEs_test,
             label="Test RMSE",
             color = "orange",
             zorder = 5)

    plt.xticks(fontsize = font_small)
    plt.yticks(fontsize = font_small)

    plt.xlim(left = 0)

    axes.legend(fontsize = font_small)
    plt.xlabel('\nK', fontsize = font_medium)
    plt.ylabel('RMSE\n', fontsize = font_medium)
    plt.title('Impact of K on the RMSE of ' + gd_type.upper() + '\n', fontsize = font_large)

    if save_plots:
        plt.savefig("graphs/impact_k_rmse_" + gd_type + ".png", bbox_inches='tight', dpi=300)
    plt.show()
    
    return RMSEs, RMSEs_test, Qs, Ps


def analyze_accuracy_gradient_descent(A_train,
                                      A_test,
                                      gd_type='sgd',
                                      k = 10, 
                                      num_epochs=5,
                                      lambda_=1e-2,
                                      eta=1e-5,
                                      save_plots = False,
                                      **kwargs):
    """
    Helper function to analyze how the training and test accuracy evolves throughout the epochs of performing gradient descent.
    The performance will be measured by computing the RMSE after each epoch on the training set and a validation set.
    
    :param A_train: a sparse training matrix to compute the in-sample error
    :param A_test: a sparse test matrix to compute the out-of-sample error
    :param gd_type: the type of gradient descent to perform (expected to be either 'sgd' or 'bgd')
    :param num_epochs: the number of epochs that should be executed by gradient descent
    :param k: the number of singular values and corresponding vectors to compute in the SVD decomposition
    :param lambda_: the regularization parameter that should be used in the gradient descent objective function
    :param eta: the learning rate that should be used in gradient descent
    :param save_plots: if True plots of the analysis results will be saved in a local directory. 
    :param kwargs: any additional parameters that should be passed to the gradient descent function
    
    :return: two lists containing the RMSE results per epoch on the training set and the test set respectively
             and the final obtained estimation of Q and P
    """
            
    # compute the largest k singular values and corresponding vectors of SVD on the training set
    U, s, V_t = linalg.svds(A_train, k, which='LM')

    # pre-multiply the components to obtain initial Q and P.T,
    # Q and P.T are smaller than A and can be stored as regular numpy arrays
    Q = np.array(U, dtype=np.float64)
    P_t = np.array(np.diag(s) @ V_t, dtype=np.float64)

    if gd_type == 'sgd':
        # perform stochastic gradient descent
        Q_new, P_new, RMSEs, RMSEs_test = stochastic_gradient_descent(A_train, Q, P_t,
                                                                      num_epochs=num_epochs, eta=eta,
                                                                      lambda_=lambda_, include_RMSE=True,
                                                                      A_test=A_test, **kwargs)
    else:
        # perform batch gradient descent
        Q_new, P_new, RMSEs, RMSEs_test = batch_gradient_descent(A_train, Q, P_t,
                                                                 num_epochs=num_epochs, eta=eta,
                                                                 lambda_=lambda_, include_RMSE=True,
                                                                 A_test=A_test, **kwargs)
    
    # Fonts that are used in the plots
    font_small = 13
    font_medium = 14
    font_large = 15
    
    fig, axes = plt.subplots(figsize = (11, 9))

    # set x-axis ticks (one per epoch)
    x_ticks = np.arange(0, num_epochs+1, 1, dtype=int)
    x_tick_labels = [str(i) for i in x_ticks]
    axes.set_xticks(x_ticks)
    axes.set_xticklabels(x_tick_labels)

    # plot the results
    plt.plot(np.arange(num_epochs+1), 
             RMSEs,
             label="Training RMSE",
             color = "blue",
             zorder = 1)
    plt.plot(np.arange(num_epochs+1), 
             RMSEs_test,
             label="Test RMSE",
             color = "orange",
             zorder = 5)

    plt.xticks(fontsize = font_small)
    plt.yticks(fontsize = font_small)

    plt.xlim(left = 0)

    axes.legend(fontsize = font_small)
    plt.xlabel('\nEpoch', fontsize = font_medium)
    plt.ylabel('RMSE\n', fontsize = font_medium)
    plt.title('Evolution of RMSE throughout the epochs using ' + gd_type.upper() + '\n', fontsize = font_large)

    if save_plots:
        plt.savefig("graphs/evolution_rmse_" + gd_type + ".png", bbox_inches='tight', dpi=300)
    plt.show()
    
    return RMSEs, RMSEs_test, Q_new, P_new


def tune_gradient_descent(A_train,
                          A_test,
                          gd_type='sgd',
                          num_epochs=[10],
                          ks = [25, 50, 75, 100], 
                          lambdas=[5e-1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                          etas=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]):
    """
    Helper function to finetune some gradient descent hyperparameters.
    RMSE is computed for each combination and the results are returned for further analysis.

    :param A_train: a sparse training matrix to compute the in-sample error
    :param A_test: a sparse test matrix to compute the out-of-sample error
    :param gd_type: the type of gradient descent to perform (expected to be either 'sgd' or 'bgd')
    :param num_epochs: the range of the number of epochs to investigate
    :param ks: the range of k values to investigate
    :param lambdas: the range of values for the regularization parameter to investigate
    :param etas: the range of values for the learning rate to investigate
    
    :return: list of dicts with each dict containing the used hyperparameter values and the final RMSE that was obtained
    """
    
    # create the parameter grid to explore
    tuning_grid = list(ParameterGrid({'epochs': num_epochs, 'eta': etas, 'lambda_' : lambdas}))
    
    results = []
    for k in ks:
        print("k: " + str(k))
        
        # compute the largest k singular values and corresponding vectors of SVD on the training set
        U, s, V_t = linalg.svds(A_train, k, which='LM')
        
        # pre-multiply the components to obtain initial Q and P.T,
        # Q and P.T are smaller than A and can be stored as regular numpy arrays
        Q = np.array(U, dtype=np.float64)
        P_t = np.array(np.diag(s) @ V_t, dtype=np.float64)
        
        for tuning_values in tuning_grid:
            if gd_type == 'sgd':
                # perform stochastic gradient descent
                Q_new, P_new, _, _ = stochastic_gradient_descent(A_train, Q, P_t,
                                                                 **tuning_values)
            else:
                # perform batch gradient descent
                Q_new, P_new, _, _ = batch_gradient_descent(A_train, Q, P_t,
                                                            **tuning_values)
            
            # compute the final RMSE that was obtained using this hyperparameter combination
            RMSE_test = compute_RMSE(A_test, Q_new, P_new)
            
            # store the results
            result = tuning_values.copy()
            result["k"] = k
            result["rmse"] = RMSE_test
            
            results.append(result)

    return results


if __name__ == "__main__":

    # Task 1: Loading the Dataset
    print("STARTING TASK 1")
    # parse the full dataset into 2 sparse matrices
    mu_sparse_matrix, um_sparse_matrix, movie_id_dict, user_id_dict = parse_dataset(subset_prob=None)
    
    # we will also already parse the probe.txt file to split the dataset into training and testing sets,
    # this is actually part of Task 4 but closely related to loading the dataset

    # parse probe.txt to extract the indices that represent ratings belonging to the test set
    movie_idxs, user_idxs = parse_test_file(movie_id_dict, user_id_dict)

    # split the sparse matrices into training and test sets (circa 99% of the data and 1% of the data respectively)
    mu_train_probe, mu_test_probe = train_test_split(mu_sparse_matrix, movie_idxs, user_idxs)
    um_train_probe, um_test_probe = train_test_split(um_sparse_matrix, user_idxs, movie_idxs)

    # split our training set further into an 80% training set and 20% validation set
    mu_train_80, mu_val_20 = train_test_relative_split(mu_train_probe, relative_test_size=0.20, random_seed=2022)
    um_train_80, um_val_20 = train_test_relative_split(um_train_probe, relative_test_size=0.20, random_seed=2022)


    # Task 2: DIMSUM
    print("STARTING TASK 2")
    # the tall skinny sparse matrix should be used, i.e., user rows and movie columns
    #A = um_sparse_matrix
    # since DIMSUM is too slow to execute on the entire dataset, we used the smaller probe test set instead
    A = um_test_probe
    
    # we scale the entries of A to be in [-1, 1], as mentioned in the original DIMSUM paper 
    # (or [0, 1] in our case as there are no negative ratings)
    A = A / A.max()
    
    # we compute the maximum gamma value (which would emit all values),
    # higher gamma values are possible but should give the same result
    col_norms = compute_col_norms(A)
    largest_col_norm = sorted(col_norms, reverse=True)[0]
    max_gamma = largest_col_norm**2

    # We will perform an experimental analysis by computing the MSE between our approximation of A.T @ A for 20 different gamma values,
    # between max_gamma/20 and max_gamma.
    # We will repeat the approximation for 10 runs in order to obtain more reliable means and standard deviations.
    mse_results, _ = analyze_mse_gamma(A, num_runs = 10, gammas = np.linspace(max_gamma/20, math.ceil(max_gamma), 20), save_plots = False)


    # Task 3: (Stochastic) Gradient Descent with Latent Factors + Task 4: Accuracy
    print("STARTING TASKS 3 & 4")

    # the wide short sparse matrix should be used, i.e., movie rows and user columns
    A_train = mu_train_probe
    A_test = mu_test_probe

    # specify the number of epochs to perform,
    # we use a limited number of epochs due to the large amount of data and relatively long runtime it requires,
    # in the case of stochastic gradient descent this limited number of epochs actually also consists of a large number of updates as well
    # (and near-convergence seems to generally happen already after 2 epochs)
    num_epochs = 2

    # specify the regularization parameter,
    # this value has been obtained via experimentation using the function tune_gradient_descent on different subsets of the original dataset
    lambda_ = 0.01

    # specify the learning rate,
    # this value has been obtained via experimentation using the function tune_gradient_descent on different subsets of the original dataset
    eta_sgd = 1e-5

    # specify the k values to analyze (1, 10, 20, ..., 90, 100)
    ks = [1]
    ks.extend(list(range(10, 101, 10)))

    # analyze k for stochastic gradient descent
    analyze_k_gradient_descent(A_train, A_test, gd_type='sgd', ks = ks, 
                               num_epochs=num_epochs, lambda_=lambda_, eta=eta_sgd,
                               save_plots = False)

    # specify the number of singular values to compute for the accuracy evolution plots,
    # the k we used was the best k coming from the prior analysis
    k = 50

    # specify the number of epochs to perform to analyze the evolution of the accuracy
    num_epochs = 5

    # BGD needed some tuning of the learning rate to obtain a usable result, so another eta was used for this analysis
    eta_bgd = 0.1/A_train.nnz # this comes down to using the gradient averaged over all samples (higher eta's seem problematic for BGD on the full dataset)

    # analyze the evolution of the in-sample and out-of-sample accuracy over multiple epochs for stochastic gradient descent
    analyze_accuracy_gradient_descent(A_train, A_test, gd_type='sgd',
                                      k = k, num_epochs=num_epochs,
                                      lambda_=lambda_, eta=eta_sgd,
                                      save_plots = False)

    # analyze the evolution of the in-sample and out-of-sample accuracy over multiple epochs for batch gradient descent
    analyze_accuracy_gradient_descent(A_train, A_test, gd_type='bgd',
                                      k = k, num_epochs=num_epochs,
                                      lambda_=lambda_, eta=eta_bgd,
                                      save_plots = False)

