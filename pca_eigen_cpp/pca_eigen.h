#ifndef PCA_EIGEN_H
#define PCA_EIGEN_H

#include <Eigen/Dense>
#include <Eigen/Eigenvalues> // For SelfAdjointEigenSolver
#include <vector>
#include <algorithm> // For std::sort, std::iota, std::reverse
#include <iostream>  // For printing (optional, for debugging)
#include <numeric>   // For std::iota
#include <string>    // Required for std::string in print_eigen_matrix

// Structure to hold PCA results
struct PCAResults {
    Eigen::MatrixXd transformed_data;
    Eigen::MatrixXd principal_components; // Eigenvectors (features x n_components)
    Eigen::VectorXd eigenvalues_k;        // Top k eigenvalues (optional, good to have)

    // Default constructor for empty results on error
    PCAResults() : transformed_data(0,0), principal_components(0,0), eigenvalues_k(0) {}
};

// Function to perform PCA
PCAResults pca_eigen(const Eigen::MatrixXd& data, int n_components);

// Helper to print Eigen matrices (optional, for debugging)
void print_eigen_matrix(const Eigen::MatrixXd& matrix, const std::string& name);
void print_eigen_vector(const Eigen::VectorXd& vec, const std::string& name);

#endif // PCA_EIGEN_H
