#include <stdio.h>
#include <stdlib.h>
#include <time.h> // Required for time() if seeding RNG that way
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h> // For random data generation
#include "pca.h"     // Contains declarations for pca() and print_matrix()

int main(void) {
    // --- 1. Define Data Parameters ---
    const size_t N_SAMPLES = 10;
    const size_t N_FEATURES = 4;
    const int N_COMPONENTS = 2; // Desired number of principal components

    if (N_COMPONENTS > N_FEATURES) {
        fprintf(stderr, "Error: Number of components cannot exceed number of features.\n");
        return EXIT_FAILURE;
    }

    // --- 2. Create and Populate Sample Data ---
    gsl_matrix *data_matrix = gsl_matrix_alloc(N_SAMPLES, N_FEATURES);
    if (data_matrix == NULL) {
        fprintf(stderr, "Failed to allocate memory for data matrix.\n");
        return EXIT_FAILURE;
    }

    // Use GSL's random number generator to fill the matrix for variety
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    if (rng == NULL) {
        fprintf(stderr, "Failed to allocate GSL RNG.\n");
        gsl_matrix_free(data_matrix);
        return EXIT_FAILURE;
    }
    // Seed the RNG (optional, for reproducible "random" data)
    // For true randomness across runs:
    // gsl_rng_set(rng, (unsigned long int)time(NULL));
    // For reproducible "random" data during testing:
    gsl_rng_set(rng, 12345);


    for (size_t i = 0; i < N_SAMPLES; ++i) {
        for (size_t j = 0; j < N_FEATURES; ++j) {
            // Generate random numbers between 0 and 10, for example
            gsl_matrix_set(data_matrix, i, j, gsl_rng_uniform(rng) * 10.0);
        }
    }
    gsl_rng_free(rng);

    printf("--- Original Data (first 5 rows if more) ---\n");
    // Print only a part of the matrix if it's too large
    size_t rows_to_print = N_SAMPLES > 5 ? 5 : N_SAMPLES;
    gsl_matrix_view data_subview = gsl_matrix_submatrix(data_matrix, 0, 0, rows_to_print, N_FEATURES);
    print_matrix(&data_subview.matrix, "Original Data Subview");
    printf("\n");


    // --- 3. Call PCA ---
    gsl_matrix *principal_components = NULL; // Will be allocated by pca()
    gsl_matrix *transformed_data = NULL;   // Will be allocated by pca()

    printf("--- Performing PCA to reduce to %d components ---\n", N_COMPONENTS);
    transformed_data = pca(data_matrix, N_COMPONENTS, &principal_components);

    // --- 4. Print Results ---
    if (transformed_data != NULL && principal_components != NULL) {
        printf("\n--- Selected Principal Components (%zu features x %d components) ---\n", principal_components->size1, N_COMPONENTS);
        print_matrix(principal_components, "Principal Components");

        printf("\n--- Transformed Data (first 5 rows if more, %zu samples x %d components) ---\n", transformed_data->size1, N_COMPONENTS);
        rows_to_print = transformed_data->size1 > 5 ? 5 : transformed_data->size1;
        // Ensure n_components for subview is correct (it's N_COMPONENTS)
        gsl_matrix_view transformed_subview = gsl_matrix_submatrix(transformed_data, 0, 0, rows_to_print, N_COMPONENTS);
        print_matrix(&transformed_subview.matrix, "Transformed Data Subview");

        printf("\nPCA completed successfully.\n");
    } else {
        fprintf(stderr, "PCA computation failed.\n");
    }

    // --- 5. Free Memory ---
    gsl_matrix_free(data_matrix);
    gsl_matrix_free(principal_components); // Safe even if NULL
    gsl_matrix_free(transformed_data);   // Safe even if NULL

    if (transformed_data == NULL || principal_components == NULL) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
