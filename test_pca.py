import unittest
import numpy as np
from pca import pca, standardize_data

class TestPCA(unittest.TestCase):

    def test_standardize_data(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        standardized = standardize_data(data)
        # Check if mean is close to 0 and std dev is close to 1 for each feature
        self.assertTrue(np.allclose(np.mean(standardized, axis=0), 0))
        self.assertTrue(np.allclose(np.std(standardized, axis=0), 1))

    def test_pca_simple(self):
        # Simple dataset where PCA should be straightforward
        data = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        n_components = 1
        transformed_data, components = pca(data, n_components)

        self.assertEqual(transformed_data.shape, (data.shape[0], n_components))
        self.assertEqual(components.shape, (data.shape[1], n_components))

        # For this simple linearly correlated data, the first principal component
        # should capture most of the variance.
        # The exact values depend on the eigenvector scaling, but the direction is key.
        # We expect the component to be roughly proportional to [1, 1] / sqrt(2)
        # or [-1, -1] / sqrt(2)
        # The principal component should align with the direction of maximum variance
        # For data y = x + c, after centering, it's y = x. So direction is [1, 1].
        # Normalized, this is [1/sqrt(2), 1/sqrt(2)] or [-1/sqrt(2), -1/sqrt(2)]
        expected_component_abs = np.abs(np.array([1, 1]) / np.sqrt(2))
        self.assertTrue(np.allclose(np.abs(components[:, 0]), expected_component_abs),
                        f"Components: {components[:,0]}, Expected (abs): {expected_component_abs}")

        # Test with n_components = 2 (should return both components)
        n_components = 2
        _, components_2 = pca(data, n_components)
        self.assertEqual(components_2.shape, (data.shape[1], n_components))
        # Components should be orthogonal
        self.assertTrue(np.isclose(np.dot(components_2[:, 0], components_2[:, 1]), 0))

        # The two components should be along [1,1] and [-1,1] (or vice-versa, signs might flip)
        comp1_abs = np.abs(components_2[:, 0])
        comp2_abs = np.abs(components_2[:, 1])

        expected_comp1_abs = np.array([1,1])/np.sqrt(2)
        expected_comp2_abs = np.array([-1,1])/np.sqrt(2) # or [1,-1]

        # Check if one component matches expected_comp1_abs and the other matches expected_comp2_abs (in terms of absolute values)
        # This is because the order of eigenvectors with same eigenvalue (or close) is not guaranteed,
        # though for this data, eigenvalues should be distinct.
        # A simpler check is that their absolute values match one of the expected directions.

        # Check that the directions are [1,1] and [-1,1] or [1,-1] up to sign and order.
        # This means their absolute dot products with these directions should be 1.
        dir1 = np.array([1,1])/np.sqrt(2)
        dir2 = np.array([-1,1])/np.sqrt(2)

        # Check if the returned components align with these directions
        c1_aligns_d1 = np.isclose(np.abs(np.dot(components_2[:, 0], dir1)), 1)
        c1_aligns_d2 = np.isclose(np.abs(np.dot(components_2[:, 0], dir2)), 1)
        c2_aligns_d1 = np.isclose(np.abs(np.dot(components_2[:, 1], dir1)), 1)
        c2_aligns_d2 = np.isclose(np.abs(np.dot(components_2[:, 1], dir2)), 1)

        self.assertTrue((c1_aligns_d1 and c2_aligns_d2) or (c1_aligns_d2 and c2_aligns_d1),
                        f"Components {components_2} do not align with expected directions {dir1}, {dir2}")


    def test_pca_dimensionality_reduction(self):
        data = np.random.rand(100, 5) # 100 samples, 5 features
        n_components = 3
        transformed_data, components = pca(data, n_components)

        self.assertEqual(transformed_data.shape, (100, n_components))
        self.assertEqual(components.shape, (5, n_components))

        # Eigenvectors (components) should be orthogonal
        for i in range(n_components):
            for j in range(i + 1, n_components):
                self.assertTrue(np.isclose(np.dot(components[:, i], components[:, j]), 0),
                                f"Components {i} and {j} are not orthogonal")

    def test_pca_zero_variance_feature(self):
        # Data with one feature having zero variance
        data = np.array([[1, 2, 0], [2, 3, 0], [3, 4, 0], [4, 5, 0], [5, 6, 0]])
        n_components = 1
        transformed_data, components = pca(data, n_components)

        self.assertEqual(transformed_data.shape, (data.shape[0], n_components))
        self.assertEqual(components.shape, (data.shape[1], n_components))
        # The component corresponding to zero variance should be zero or near zero
        # The pca implementation handles this via standardize_data, where std_dev = 1 for zero variance features.
        # The eigenvector for the zero variance feature should ideally be [0,0,1] or similar,
        # and its eigenvalue should be 0.
        # After sorting, this component should not be selected if n_components < number of original features.

        # Let's check that the transformed data has the correct shape
        # and that the components have the correct shape.
        # More specific checks would require knowing the expected eigenvalues/vectors.
        self.assertIsNotNone(transformed_data)
        self.assertIsNotNone(components)

        # Check if the principal component for the zero-variance feature is indeed [0,0,1] or similar,
        # meaning it aligns with that axis.
        # Eigenvalues are sorted, so the component with eigenvalue near zero should be last.
        # If n_components = 1, it should pick the component with largest variance.

        # Let's re-run with n_components = 3 to see all components
        _, components_all = pca(data, 3)
        eigenvalues = np.linalg.eigvals(np.cov(standardize_data(data), rowvar=False))
        sorted_eigenvalues_indices = np.argsort(eigenvalues)[::-1]

        # The eigenvector corresponding to the smallest eigenvalue (for the zero var feature)
        # should ideally be [0,0,c] or [0,0,-c]
        # This means its first two elements are close to zero.
        # The third eigenvector (corresponding to the smallest eigenvalue)
        # might be components_all[:, sorted_eigenvalues_indices[-1]]
        # However, np.linalg.eig might not return them perfectly sorted if eigenvalues are very close (e.g. 0)
        # pca sorts them. So components_all[:, 2] is the one for the smallest eigenvalue.

        # The third component (associated with the zero variance feature) should be orthogonal
        # to the first two components of the *original* non-zero-variance data space.
        # And its primary contribution should be along the 3rd feature axis.
        # This means components_all[2,2] (3rd feature of 3rd component) should be close to 1 or -1.
        # And components_all[0,2] and components_all[1,2] should be close to 0.
        # This test depends on the fact that standardize_data replaces std_dev=0 with 1,
        # so the covariance matrix will have zeros for the third feature, except its diagonal.
        # The eigenvalues would be var(feat1_std), var(feat2_std), 0.
        # The eigenvector for eigenvalue 0 should be [0,0,1].
        # For data's covariance matrix [[1,1,0],[1,1,0],[0,0,0]], eigenvalues are [2,0,0].
        # pca sorts eigenvalues in descending order.
        # components_all[:,0] corresponds to eigenvalue 2.
        # components_all[:,1] and components_all[:,2] correspond to the eigenvalue 0.
        # One of these (for L=0) should be [0,0,+-1] (representing the original zero-variance axis).
        # The other (for L=0) should be orthogonal to it and to the first component, e.g. [+-1/sqrt(2), -+1/sqrt(2), 0].

        # Target vector representing the axis of the original zero-variance feature
        z_axis_vector_abs = np.array([0.0, 0.0, 1.0])

        # Check if the absolute values of components_all[:,1] match z_axis_vector_abs
        comp1_is_z_aligned = np.allclose(np.abs(components_all[:, 1]), z_axis_vector_abs)
        # Check if the absolute values of components_all[:,2] match z_axis_vector_abs
        comp2_is_z_aligned = np.allclose(np.abs(components_all[:, 2]), z_axis_vector_abs)

        self.assertTrue(comp1_is_z_aligned or comp2_is_z_aligned,
                        f"Neither of the components for eigenvalue 0 is aligned with the z-axis (expected [0,0,1] or [0,0,-1] in absolute terms).\n"
                        f"Components (abs) for eigenvalue 0 are:\n{np.abs(components_all[:, 1])}\nand\n{np.abs(components_all[:, 2])}")

        # Further check: if n_components=1 was chosen, it should pick components_all[:,0]
        # (which is related to the non-zero variance part).
        # The initial part of this test already called pca(data,1) and stored in 'components'.
        # components_n1 should be components_all[:,0]
        self.assertTrue(np.allclose(np.abs(components[:,0]), np.abs(components_all[:,0])),
                        "PCA with n_components=1 did not pick the first component from n_components=3")
        # And this component should not be the z-axis one.
        self.assertFalse(np.allclose(np.abs(components[:,0]), z_axis_vector_abs),
                         "PCA with n_components=1 picked the z-axis component for zero variance feature.")


    def test_pca_edge_case_n_components_equals_features(self):
        data = np.random.rand(20, 4) # 20 samples, 4 features
        n_components = 4
        transformed_data, components = pca(data, n_components)

        self.assertEqual(transformed_data.shape, (data.shape[0], n_components))
        self.assertEqual(components.shape, (data.shape[1], n_components))
        # Components should be orthogonal
        for i in range(n_components):
            for j in range(i + 1, n_components):
                self.assertTrue(np.isclose(np.dot(components[:, i], components[:, j]), 0),
                                f"Components {i} and {j} are not orthogonal")

    def test_pca_edge_case_n_components_one_with_more_features(self):
        data = np.random.rand(30, 5) # 30 samples, 5 features
        n_components = 1
        transformed_data, components = pca(data, n_components)

        self.assertEqual(transformed_data.shape, (data.shape[0], n_components))
        self.assertEqual(components.shape, (data.shape[1], n_components))

    def test_variance_retention(self):
        data = np.array([
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [5, 6, 7, 8]
        ]) # Data with clear variance directions
        n_features = data.shape[1]
        transformed_data_full, components_full = pca(data, n_features)

        # Calculate variance along each principal component
        variances = np.var(transformed_data_full, axis=0)

        # Check that variances are sorted in descending order
        # (allowing for small numerical precision issues if they are very close, e.g. for zero eigenvalues)
        for i in range(len(variances) - 1):
            # self.assertGreaterEqual(variances[i], variances[i+1]) # This can be too strict for ~0 values
            self.assertTrue(variances[i] >= variances[i+1] - 1e-9, # Allow small tolerance
                                    f"Variance of PC{i} ({variances[i]}) is not greater/equal than PC{i+1} ({variances[i+1]})")


    def test_known_results_orthogonal_data(self):
        # Data where features are already orthogonal and centered (almost)
        # PCA should identify these axes.
        data = np.array([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 2, 0],
            [0, -2, 0],
            [0, 0, 3],
            [0, 0, -3]
        ])
        # standardize_data will center it further and scale by std dev.
        # std_devs will be [1, 2, 3] for the uncentered data.
        # After standardization, data will be:
        # [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]] (approx)
        # Covariance matrix should be identity matrix. Eigenvalues all 1. Eigenvectors are standard axes.

        n_components = 3
        _, components = pca(data, n_components)

        # The components should be the standard basis vectors (e1, e2, e3), possibly permuted or with sign flips.
        # This means each column in `components` should be a vector like [1,0,0] or [0,-1,0] etc.
        # Check that each component vector is a standard basis vector (abs value)
        # And that they are orthogonal (already covered by dot product checks)

        # Check that the columns of components are permutations of e_i (up to sign)
        # Each column should have one element close to 1 (or -1) and others close to 0.
        for i in range(n_components):
            col_abs = np.abs(components[:, i])
            # Check that one element is close to 1 and others are close to 0
            self.assertTrue(np.sum(np.isclose(col_abs, 1.0)) == 1, f"Component {i} ({components[:,i]}) is not a standard basis vector (abs val check)")
            self.assertTrue(np.sum(np.isclose(col_abs, 0.0)) == n_components - 1, f"Component {i} ({components[:,i]}) is not a standard basis vector (zeros check)")

        # Ensure orthogonality (already in other tests, but good for specific case)
        for i in range(n_components):
            for j in range(i + 1, n_components):
                self.assertTrue(np.isclose(np.dot(components[:, i], components[:, j]), 0),
                                f"Components {i} and {j} are not orthogonal for known data")


if __name__ == '__main__':
    unittest.main()
