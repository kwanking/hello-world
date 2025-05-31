import numpy as np

def standardize_data(X):
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    # Avoid division by zero if a feature has zero variance
    std_dev[std_dev == 0] = 1
    return (X - mean) / std_dev

def pca(data, n_components):
    """
    Performs Principal Component Analysis (PCA) on the input data.

    Args:
        data (np.ndarray): Input data, where rows are samples and columns are features.
        n_components (int): Number of principal components to keep.

    Returns:
        tuple: A tuple containing:
            - transformed_data (np.ndarray): Data projected onto the principal components.
            - components (np.ndarray): Selected principal components (eigenvectors).
    """
    # Standardize the data
    standardized_data = standardize_data(data)

    # Calculate the covariance matrix
    covariance_matrix = np.cov(standardized_data, rowvar=False)

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select top k eigenvectors
    components = sorted_eigenvectors[:, :n_components]

    # Project the data
    transformed_data = np.dot(standardized_data, components)

    return transformed_data, components
