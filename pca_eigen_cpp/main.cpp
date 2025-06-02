#include <iostream>
#include "pca_eigen.h" // Our PCA implementation
#include <Eigen/Dense>   // For Eigen::MatrixXd

int main() {
    // --- 1. Create Sample Data ---
    Eigen::MatrixXd data(5, 4); // 5 samples, 4 features
    data << 1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0;
    // A simple, structured dataset to make outputs somewhat predictable

    // Add some variation to make PCA more interesting
    data(0,0) = 2.5;
    data(1,1) = 5.5;
    data(2,2) = 10.5;
    data(3,3) = 16.5;
    data(4,0) = 15.0;


    std::cout << "--- Original Data ---" << std::endl;
    print_eigen_matrix(data, "Original Data");
    // std::cout << std::endl; // print_eigen_matrix already adds a newline

    // --- 2. Define Number of Components ---
    int n_components = 2;
    if (n_components <=0) {
        std::cerr << "Error: n_components must be positive." << std::endl;
        return 1;
    }
    if (n_components > data.cols()) {
        std::cerr << "Error: n_components (" << n_components
                  << ") cannot be greater than the number of features (" << data.cols()
                  << ")." << std::endl;
        return 1;
    }
    std::cout << "--- Performing PCA to reduce to " << n_components << " components ---" << std::endl << std::endl;

    // --- 3. Call PCA ---
    PCAResults results = pca_eigen(data, n_components);

    // --- 4. Print Results ---
    if (results.transformed_data.size() == 0 || results.principal_components.size() == 0) {
        // pca_eigen prints internal errors to cerr, so just a general message here
        std::cerr << "PCA computation failed or returned empty results. Exiting." << std::endl;
        return 1;
    }

    std::cout << "--- Top " << n_components << " Eigenvalues ---" << std::endl;
    print_eigen_vector(results.eigenvalues_k, "Selected Eigenvalues");
    // std::cout << std::endl; // print_eigen_vector already adds a newline

    std::cout << "--- Principal Components (Eigenvectors) ---" << std::endl;
    print_eigen_matrix(results.principal_components, "Principal Components");
    // std::cout << std::endl; // print_eigen_matrix already adds a newline

    std::cout << "--- Transformed Data ---" << std::endl;
    print_eigen_matrix(results.transformed_data, "Transformed Data");
    // std::cout << std::endl; // print_eigen_matrix already adds a newline

    std::cout << "PCA using Eigen completed successfully." << std::endl;

    return 0;
}
