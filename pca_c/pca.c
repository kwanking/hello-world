#include "pca.h"
#include <stdio.h>
#include <gsl/gsl_statistics.h>      // Changed from gsl_statistics_double.h
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>           // Added for eigenvalue computation
#include <math.h> // For fabs in SD check, though GSL SD should be non-negative

// Function to print a GSL matrix (for debugging)
void print_matrix(const gsl_matrix *m, const char* name) {
    if (name) {
        printf("Matrix %s (size %zu x %zu):\n", name, m->size1, m->size2);
    } else {
        printf("Matrix (size %zu x %zu):\n", m->size1, m->size2);
    }
    for (size_t i = 0; i < m->size1; i++) {
        for (size_t j = 0; j < m->size2; j++) {
            printf("%g ", gsl_matrix_get(m, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

// Function to print a GSL vector (for debugging)
void print_vector(const gsl_vector *v, const char* name) {
    if (name) {
        printf("Vector %s (size %zu):\n", name, v->size);
    } else {
        printf("Vector (size %zu):\n", v->size);
    }
    for (size_t i = 0; i < v->size; i++) {
        printf("%g ", gsl_vector_get(v, i));
    }
    printf("\n\n");
}

// Function to calculate the mean of each column in a matrix
gsl_vector* calculate_column_means(const gsl_matrix *m) {
    if (!m) return NULL;
    size_t num_cols = m->size2;
    gsl_vector *means = gsl_vector_alloc(num_cols);
    if (!means) {
        fprintf(stderr, "Error: Failed to allocate memory for means vector.\n");
        return NULL;
    }

    for (size_t j = 0; j < num_cols; ++j) {
        gsl_vector_const_view col_view = gsl_matrix_const_column(m, j);
        double mean = gsl_stats_mean(col_view.vector.data, col_view.vector.stride, col_view.vector.size);
        gsl_vector_set(means, j, mean);
    }
    return means;
}

// Function to calculate the standard deviation of each column
gsl_vector* calculate_column_sds(const gsl_matrix *m, const gsl_vector *means) {
    if (!m || !means || m->size2 != means->size) return NULL;
    size_t num_cols = m->size2;
    gsl_vector *sds = gsl_vector_alloc(num_cols);
    if (!sds) {
        fprintf(stderr, "Error: Failed to allocate memory for sds vector.\n");
        return NULL;
    }

    for (size_t j = 0; j < num_cols; ++j) {
        gsl_vector_const_view col_view = gsl_matrix_const_column(m, j);
        double mean_j = gsl_vector_get(means, j);
        // Use gsl_stats_sd_m for standard deviation with a known mean.
        double sd = gsl_stats_sd_m(col_view.vector.data, col_view.vector.stride, col_view.vector.size, mean_j);
        gsl_vector_set(sds, j, sd);
    }
    return sds;
}

// Function to standardize data (subtract mean, divide by standard deviation)
gsl_matrix* standardize_data(const gsl_matrix *data) {
    if (!data) return NULL;
    gsl_vector *means = calculate_column_means(data);
    if (!means) return NULL;
    gsl_vector *sds = calculate_column_sds(data, means);
    if (!sds) {
        gsl_vector_free(means);
        return NULL;
    }

    gsl_matrix *standardized_matrix = gsl_matrix_alloc(data->size1, data->size2);
    if (!standardized_matrix) {
        fprintf(stderr, "Error: Failed to allocate memory for standardized matrix.\n");
        gsl_vector_free(means);
        gsl_vector_free(sds);
        return NULL;
    }

    for (size_t j = 0; j < data->size2; ++j) { // For each column
        double mean_j = gsl_vector_get(means, j);
        double sd_j = gsl_vector_get(sds, j);

        for (size_t i = 0; i < data->size1; ++i) { // For each row
            double val = gsl_matrix_get(data, i, j);
            if (sd_j != 0.0 && fabs(sd_j) > 1e-15) { // Check sd is not effectively zero
                 gsl_matrix_set(standardized_matrix, i, j, (val - mean_j) / sd_j);
            } else {
                 gsl_matrix_set(standardized_matrix, i, j, 0.0); // Set to 0 if sd is zero
            }
        }
    }

    gsl_vector_free(means);
    gsl_vector_free(sds);
    return standardized_matrix;
}

// Function to calculate the covariance matrix
// Assumes input 'm_standardized' is already standardized data (mean-centered).
gsl_matrix* calculate_covariance_matrix(const gsl_matrix *m_standardized) {
    if (!m_standardized) return NULL;
    size_t N = m_standardized->size1; // Number of samples
    size_t P = m_standardized->size2; // Number of features

    if (N <= 1) {
        fprintf(stderr, "Error: Need at least 2 samples to calculate covariance matrix.\n");
        return NULL;
    }

    gsl_matrix *covariance_matrix = gsl_matrix_alloc(P, P);
    if (!covariance_matrix) {
        fprintf(stderr, "Error: Failed to allocate memory for covariance matrix.\n");
        return NULL;
    }

    // C = (1/(N-1)) * m_standardized^T * m_standardized
    // gsl_blas_dgemm computes C = alpha*A*B + beta*C
    // Here, A = m_standardized^T, B = m_standardized
    // alpha = 1.0 / (N - 1)
    // beta = 0.0
    // Operation for A: CblasTrans (Transpose)
    // Operation for B: CblasNoTrans (No Transpose)
    gsl_blas_dgemm(CblasTrans, CblasNoTrans,
                   1.0 / (double)(N - 1), m_standardized, m_standardized,
                   0.0, covariance_matrix);

    return covariance_matrix;
}

// Function to compute eigenvalues and eigenvectors of a symmetric matrix (covariance matrix)
void get_eigenvalues_eigenvectors(const gsl_matrix *covariance_matrix,
                                  gsl_vector **out_eigenvalues,
                                  gsl_matrix **out_eigenvectors) {
    // Ensure covariance_matrix is square
    if (covariance_matrix->size1 != covariance_matrix->size2) {
        fprintf(stderr, "Error: Covariance matrix must be square.\n");
        *out_eigenvalues = NULL;
        *out_eigenvectors = NULL;
        return;
    }

    size_t p = covariance_matrix->size1; // Number of features

    // Allocate space for eigenvalues and eigenvectors
    *out_eigenvalues = gsl_vector_alloc(p);
    *out_eigenvectors = gsl_matrix_alloc(p, p);
    if (!(*out_eigenvalues) || !(*out_eigenvectors)) {
        fprintf(stderr, "Error: Could not allocate memory for eigenvalues/eigenvectors.\n");
        if (*out_eigenvalues) gsl_vector_free(*out_eigenvalues);
        if (*out_eigenvectors) gsl_matrix_free(*out_eigenvectors);
        *out_eigenvalues = NULL;
        *out_eigenvectors = NULL;
        return;
    }

    // GSL's eigen functions can modify the input matrix, so we need to work on a copy.
    gsl_matrix *cov_matrix_copy = gsl_matrix_alloc(p, p);
    if (!cov_matrix_copy) {
        fprintf(stderr, "Error: Could not allocate memory for covariance matrix copy.\n");
        gsl_vector_free(*out_eigenvalues);
        gsl_matrix_free(*out_eigenvectors);
        *out_eigenvalues = NULL;
        *out_eigenvectors = NULL;
        return;
    }
    gsl_matrix_memcpy(cov_matrix_copy, covariance_matrix);

    // Create a workspace for eigenvalue computation
    gsl_eigen_symmv_workspace *workspace = gsl_eigen_symmv_alloc(p);
    if (!workspace) {
        fprintf(stderr, "Error: Could not allocate GSL eigenvalue workspace.\n");
        gsl_matrix_free(cov_matrix_copy);
        gsl_vector_free(*out_eigenvalues);
        gsl_matrix_free(*out_eigenvectors);
        *out_eigenvalues = NULL;
        *out_eigenvectors = NULL;
        return;
    }

    // Compute eigenvalues and eigenvectors
    // GSL stores eigenvectors as columns in the *out_eigenvectors matrix.
    int status = gsl_eigen_symmv(cov_matrix_copy, *out_eigenvalues, *out_eigenvectors, workspace);

    gsl_eigen_symmv_free(workspace);
    gsl_matrix_free(cov_matrix_copy);

    if (status != 0) {
        fprintf(stderr, "Error: GSL eigenvalue computation failed with status %d.\n", status);
        gsl_vector_free(*out_eigenvalues);
        gsl_matrix_free(*out_eigenvectors);
        *out_eigenvalues = NULL;
        *out_eigenvectors = NULL;
        return;
    }

    // Sort eigenvalues in descending order and rearrange eigenvectors accordingly
    // GSL_EIGEN_SORT_VAL_DESC sorts eigenvalues in descending order.
    // The corresponding eigenvectors are reordered to match.
    status = gsl_eigen_symmv_sort(*out_eigenvalues, *out_eigenvectors, GSL_EIGEN_SORT_VAL_DESC);

    if (status != 0) {
        fprintf(stderr, "Error: GSL eigenvalue/eigenvector sorting failed with status %d.\n", status);
        gsl_vector_free(*out_eigenvalues);
        gsl_matrix_free(*out_eigenvectors);
        *out_eigenvalues = NULL;
        *out_eigenvectors = NULL;
        return;
    }
}


gsl_matrix* pca(const gsl_matrix *data, int n_components, gsl_matrix **out_principal_components) {
    if (data == NULL || n_components <= 0) {
        fprintf(stderr, "Error: Invalid input data or n_components.\n");
        if (out_principal_components != NULL) *out_principal_components = NULL;
        return NULL;
    }
    if (n_components > data->size2) {
        fprintf(stderr, "Error: n_components cannot be greater than the number of features.\n");
        if (out_principal_components != NULL) *out_principal_components = NULL;
        return NULL;
    }
     if (out_principal_components == NULL) {
        fprintf(stderr, "Error: out_principal_components parameter cannot be NULL.\n");
        return NULL;
    }


    gsl_matrix *std_data = NULL;
    gsl_matrix *cov_matrix = NULL;
    gsl_vector *eigenvalues = NULL;
    gsl_matrix *eigenvectors = NULL; // Full set of eigenvectors
    gsl_matrix *transformed_data = NULL;
    *out_principal_components = NULL; // Initialize output parameter

    // 1. Standardize Data
    std_data = standardize_data(data);
    if (std_data == NULL) {
        fprintf(stderr, "Error: Failed to standardize data.\n");
        goto cleanup;
    }

    // 2. Calculate Covariance Matrix
    cov_matrix = calculate_covariance_matrix(std_data);
    if (cov_matrix == NULL) {
        fprintf(stderr, "Error: Failed to calculate covariance matrix.\n");
        goto cleanup;
    }

    // 3. Compute Eigenvalues/Eigenvectors
    get_eigenvalues_eigenvectors(cov_matrix, &eigenvalues, &eigenvectors);
    if (eigenvalues == NULL || eigenvectors == NULL) {
        fprintf(stderr, "Error: Failed to compute eigenvalues/eigenvectors.\n");
        goto cleanup;
    }

    // 4. Select Top k Eigenvectors (Principal Components)
    size_t p_features = data->size2;
    *out_principal_components = gsl_matrix_alloc(p_features, n_components);
    if (*out_principal_components == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for principal components matrix.\n");
        goto cleanup;
    }

    for (int j = 0; j < n_components; ++j) {
        gsl_vector_view col_view = gsl_matrix_column(eigenvectors, j);
        gsl_matrix_set_col(*out_principal_components, j, &col_view.vector);
    }

    // 5. Project Data: X_transformed = X_standardized * W
    size_t n_samples = data->size1;
    transformed_data = gsl_matrix_alloc(n_samples, n_components);
    if (transformed_data == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for transformed data matrix.\n");
        // *out_principal_components was allocated, so it needs cleanup if this step fails
        // This specific scenario is handled by the final check in cleanup if transformed_data is NULL
        goto cleanup;
    }

    int dgemm_status = gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                                     1.0, std_data, *out_principal_components,
                                     0.0, transformed_data);
    if (dgemm_status != 0) {
        fprintf(stderr, "Error: Matrix multiplication for data projection failed (dgemm status: %d).\n", dgemm_status);
        gsl_matrix_free(transformed_data);
        transformed_data = NULL;
        // *out_principal_components also needs to be freed here as it's an output param
        // and the function will return NULL overall
        gsl_matrix_free(*out_principal_components);
        *out_principal_components = NULL;
        goto cleanup; // Though not strictly necessary as we'd fall through to cleanup anyway
    }

cleanup:
    gsl_matrix_free(std_data);
    gsl_matrix_free(cov_matrix);
    gsl_vector_free(eigenvalues);
    gsl_matrix_free(eigenvectors);

    if (transformed_data == NULL) { // If any step failed and transformed_data is not set (or explicitly NULLed)
        if (out_principal_components != NULL && *out_principal_components != NULL) {
            // If out_principal_components was allocated but the process failed overall, free it.
            gsl_matrix_free(*out_principal_components);
            *out_principal_components = NULL;
        }
    }

    return transformed_data;
}
