import numpy as np
from pca import pca

def main():
    """
    Main function to demonstrate PCA.
    """
    # Create sample data (10 samples, 3 features)
    sample_data = np.random.rand(10, 3)
    # sample_data = np.array([
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9],
    #     [10, 11, 12],
    #     [2, 1, 0.5],
    #     [5, 4.5, 6.5],
    #     [8, 7.5, 9.5],
    #     [11, 10.5, 12.5],
    #     [3, 2.5, 1.5],
    #     [6, 5.5, 7.5]
    # ])


    print("Original Data (first 5 rows):")
    print(sample_data[:5])
    print(f"Shape: {sample_data.shape}\n")

    # Define the number of principal components to keep
    n_components = 2

    print(f"Number of principal components to keep: {n_components}\n")

    # Call the pca function
    transformed_data, components = pca(sample_data, n_components)

    # Print the transformed data
    print("Transformed Data (first 5 rows):")
    print(transformed_data[:5])
    print(f"Shape: {transformed_data.shape}\n")

    # Print the components (eigenvectors)
    print("Principal Components (Eigenvectors):")
    print(components)
    print(f"Shape: {components.shape}\n")

if __name__ == "__main__":
    main()
