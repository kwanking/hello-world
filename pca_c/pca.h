#ifndef PCA_H
#define PCA_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_statistics.h> // For gsl_stats_mean, gsl_stats_sd
#include <gsl/gsl_blas.h>      // For matrix multiplication if needed for covariance
#include <stdio.h>             // For printf in print_matrix/vector

// Function to print a GSL matrix (for debugging)
void print_matrix(const gsl_matrix *m, const char* name);

// Function to print a GSL vector (for debugging)
void print_vector(const gsl_vector *v, const char* name);

// Function to calculate the mean of each column in a matrix
gsl_vector* calculate_column_means(const gsl_matrix *m);

// Function to calculate the standard deviation of each column
gsl_vector* calculate_column_sds(const gsl_matrix *m, const gsl_vector *means);

// Function to standardize data (subtract mean, divide by standard deviation)
// Returns a new matrix with the standardized data.
gsl_matrix* standardize_data(const gsl_matrix *data);

// Function to calculate the covariance matrix
// Assumes input 'm' is already standardized (mean-centered for this formula, though full standardization is better).
// The formula C = (1 / (N - 1)) * M_standardized^T * M_standardized implies M_standardized is used.
gsl_matrix* calculate_covariance_matrix(const gsl_matrix *m_standardized);

// Function to compute eigenvalues and eigenvectors of a symmetric matrix (covariance matrix)
// Allocates memory for eigenvalues (vector) and eigenvectors (matrix)
// Eigenvectors are stored as columns in the output matrix.
// GSL's gsl_eigen_symmv sorts eigenvalues in ascending order by default.
// We will sort them in descending order afterwards.
#include <gsl/gsl_eigen.h> // For gsl_eigen_symmv_workspace, GSL_EIGEN_SORT_VAL_DESC etc.

void get_eigenvalues_eigenvectors(const gsl_matrix *covariance_matrix,
                                  gsl_vector **eigenvalues,  // Output: pointer to allocated gsl_vector
                                  gsl_matrix **eigenvectors); // Output: pointer to allocated gsl_matrix

// Main PCA function
// data: input data matrix (samples x features)
// n_components: number of principal components to keep
// out_principal_components: output parameter, will be allocated and filled with
//                           the top 'n_components' eigenvectors (features x n_components)
// Returns: transformed data matrix (samples x n_components), or NULL on error.
gsl_matrix* pca(const gsl_matrix *data, int n_components, gsl_matrix **out_principal_components);


#endif // PCA_H
