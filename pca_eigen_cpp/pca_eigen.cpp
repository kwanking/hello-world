#include "pca_eigen.h"
#include <stdexcept> // For runtime_error (though not used in provided snippet, good for robustness)
#include <vector>    // For std::vector
#include <numeric>   // For std::iota
#include <algorithm> // For std::sort
#include <cmath>     // For std::sqrt

void print_eigen_matrix(const Eigen::MatrixXd& matrix, const std::string& name) {
    if (!name.empty()) {
        std::cout << name << " (rows: " << matrix.rows() << ", cols: " << matrix.cols() << "):\n";
    }
    if (matrix.size() == 0) {
        std::cout << "[empty matrix]\n\n"; // Added extra newline for better separation
        return;
    }
    std::cout << matrix << std::endl << std::endl; // Added extra newline
}

void print_eigen_vector(const Eigen::VectorXd& vec, const std::string& name) {
    if (!name.empty()) {
        std::cout << name << " (size: " << vec.size() << "):\n";
    }
     if (vec.size() == 0) {
        std::cout << "[empty vector]\n\n"; // Added extra newline
        return;
    }
    std::cout << vec.transpose() << std::endl << std::endl; // Print as row vector for brevity, added newline
}

PCAResults pca_eigen(const Eigen::MatrixXd& data, int n_components) {
    PCAResults results; // Will be returned, empty by default

    if (data.rows() == 0 || data.cols() == 0) {
        std::cerr << "Error: Input data matrix is empty." << std::endl;
        return results; // Returns default (empty) PCAResults
    }
    if (n_components <= 0) {
        std::cerr << "Error: Number of components must be positive." << std::endl;
        return results;
    }
    if (n_components > data.cols()) {
        std::cerr << "Error: Number of components cannot exceed number of features." << std::endl;
        return results;
    }
     if (data.rows() <= 1 && data.cols() > 0) { // Check if only one sample or no samples for variance calculation
        std::cerr << "Error: Cannot compute variance with zero or one sample." << std::endl;
        return results;
    }


    // 1. Standardize Data
    Eigen::MatrixXd standardized_data(data.rows(), data.cols());
    Eigen::VectorXd means = data.colwise().mean(); // Column-wise means
    Eigen::VectorXd std_devs(data.cols());

    for (long j = 0; j < data.cols(); ++j) {
        double mean_val = means(j);
        // Calculate (X_i - mean)^2 sum for column j
        double sq_sum = (data.col(j).array() - mean_val).square().sum();
        // Sample standard deviation: sqrt( sum( (X_i - mean)^2 ) / (N-1) )
        double std_dev_val = std::sqrt(sq_sum / (data.rows() - 1.0));

        if (std_dev_val < 1e-9) { // Avoid division by zero or near-zero std deviation
            std_dev_val = 1.0;
            // If std_dev is effectively zero, standardized column will be all zeros
            // because (data.col(j).array() - mean_val) will be all zeros too.
            // Then dividing by 1.0 results in zeros.
        }
        std_devs(j) = std_dev_val;
        standardized_data.col(j) = (data.col(j).array() - mean_val) / std_dev_val;
    }
    // print_eigen_matrix(standardized_data, "Standardized Data");


    // 2. Calculate Covariance Matrix
    // Cov = (X_std^T * X_std) / (N-1)
    Eigen::MatrixXd cov_matrix = (standardized_data.transpose() * standardized_data) / (data.rows() - 1.0);
    // print_eigen_matrix(cov_matrix, "Covariance Matrix");

    // 3. Eigenvalue Decomposition
    // Covariance matrix is symmetric and positive semi-definite.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(cov_matrix);
    if (eigensolver.info() != Eigen::Success) {
        std::cerr << "Error: Eigenvalue decomposition failed." << std::endl;
        return results;
    }

    Eigen::VectorXd eigenvalues_all = eigensolver.eigenvalues();         // Sorted: smallest to largest by default
    Eigen::MatrixXd eigenvectors_all = eigensolver.eigenvectors();     // Columns correspond to eigenvalues

    // 4. Sort Eigenvalues/Eigenvectors (descending order of eigenvalues)
    // Eigen's SelfAdjointEigenSolver sorts eigenvalues in increasing order. We need descending.
    std::vector<int> indices(eigenvalues_all.size());
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ...

    // Sort indices based on eigenvalues in descending order
    std::sort(indices.begin(), indices.end(),
              [&eigenvalues_all](int a, int b) { return eigenvalues_all(a) > eigenvalues_all(b); });

    results.eigenvalues_k.resize(n_components);
    results.principal_components.resize(data.cols(), n_components); // Features x N_Components

    for (int k = 0; k < n_components; ++k) {
        results.eigenvalues_k(k) = eigenvalues_all(indices[k]);
        results.principal_components.col(k) = eigenvectors_all.col(indices[k]);
    }
    // print_eigen_vector(results.eigenvalues_k, "Selected Eigenvalues (Top k)");
    // print_eigen_matrix(results.principal_components, "Principal Components (Top k Eigenvectors)");


    // 5. Project Data
    // Transformed Data = Standardized Data * Principal Components
    // (N x P) * (P x k) = (N x k)
    results.transformed_data = standardized_data * results.principal_components;
    // print_eigen_matrix(results.transformed_data, "Transformed Data");

    return results;
}
