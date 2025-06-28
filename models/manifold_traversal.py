import numpy as np
import math
import time
from scipy.linalg import svd
import matplotlib.pyplot as plt

from ..utils.utils_TISVD import TISVD_gw

from .traversal_network import TraversalNetwork
from .training_results import TrainingResults


class ManifoldTraversal:
    """
    Main class for manifold traversal denoising.
    """

    def __init__(self, intrinsic_dim, ambient_dim, sigma,
                 R_denoising=0.4, R_1st_order_nbhd=0.8, R_is_const=True,
                 d_parallel=0.01414, prod_coeff=1.2, exp_coeff=0.5):
        """
        Initialize manifold traversal with hyperparameters.
        
        Args:
            intrinsic_dim:      Intrinsic dimension of the manifold (d)
            ambient_dim:        Ambient dimension of the data (D)
            sigma:              Noise standard deviation
            R_denoising:        Denoising radius
            R_1st_order_nbhd:   First-order neighborhood radius
            R_is_const:         Whether denoising radius is constant or adaptive
            d_parallel:         Parallel component parameter for adaptive radius
            prod_coeff:         Product coefficient for adaptive radius
            exp_coeff:          Exponent coefficient for adaptive radius
        """
        # hyperparameters
        self.intrinsic_dim = intrinsic_dim  # d
        self.ambient_dim = ambient_dim  # D
        self.sigma = sigma
        self.R_denoising = R_denoising
        self.R_1st_order_nbhd = R_1st_order_nbhd
        self.R_is_const = R_is_const
        self.d_parallel = d_parallel
        self.prod_coeff = prod_coeff
        self.exp_coeff = exp_coeff

        # init network and results
        self.network = TraversalNetwork()
        self.results = TrainingResults()
        self._frame_num = 0

    def fit(self, X_noisy, X_clean, batch_size=4000, verbose=True):
        """
        Train the manifold traversal network on noisy data.
        
        Args:
            X_noisy: Noisy training data of shape (ambient_dim, n_samples)
            X_clean: Clean training data of shape (ambient_dim, n_samples)  
            batch_size: Batch size for progress reporting
            verbose: Whether to print progress
            
        Returns:
            TrainingResults object containing training metrics
        """
        n_samples = X_noisy.shape[1]

        if verbose:
            print(f"Training manifold traversal on {n_samples} samples...")

        sample_idx = 0
        while sample_idx < n_samples:
            start_time = time.time()

            # process batch_size samples
            batch_end = min(sample_idx + batch_size, n_samples)

            for i in range(sample_idx, batch_end):
                x_noisy = X_noisy[:, i]
                x_clean = X_clean[:, i]

                # process single sample
                x_denoised = self._process_sample(x_noisy)

                # update results
                mt_error = np.sum((x_denoised - x_clean) ** 2)
                data_error = np.sum((x_noisy - x_clean) ** 2)
                self.results.update(mt_error, data_error)

            # record batch timing
            batch_time = time.time() - start_time
            self.results.training_times.append(batch_time)

            if verbose:
                print(f"{batch_end} samples processed (batch time: {batch_time:.2f}s)")

            sample_idx = batch_end

        if verbose:
            total_time = sum(self.results.training_times)
            print(f"Training complete! Total time: {total_time:.2f}s")
            print(f"Network statistics: {self.network.get_network_stats()}")

        return self.results

    def denoise(self, x):
        """
        Denoise a single point using the trained network.
        Args:
            x: Noisy point of shape (ambient_dim,)
        Returns:
            Denoised point of shape (ambient_dim,)
        """
        if self.network.num_landmarks == 0:
            # no network trained yet -> return input
            return x.copy()

        # perform traversal to find best landmark
        landmark, _, _, _, _ = self._perform_traversal(x)

        # denoise using local model at best landmark
        return self._denoise_local(x, landmark)

    def _process_sample(self, x):
        """Process a single training sample."""
        if self.network.num_landmarks == 0:
            # init first landmark
            self._initialize_new_landmark(x)
            return x.copy()

        # perform traversal to find best landmark
        landmark, phi, trajectory, edge_orders, mults = self._perform_traversal(x)

        # evaluate traversal result and decide action
        R_d_sq = self._compute_denoising_radius_sq(landmark)

        if phi <= R_d_sq:
            # point is inlier -> denoise and update model
            x_denoised = self._denoise_local(x, landmark)
            self._update_landmark(x, landmark)
            return x_denoised
        else:
            # point is outlier -> try exhaustive search
            best_landmark, best_phi = self._exhaustive_search(x)
            R_d_sq_best = self._compute_denoising_radius_sq(best_landmark)

            if best_phi <= R_d_sq_best:
                # found suitable landmark via exhaustive search
                # add zero-order edge if not self-loop
                if landmark is not best_landmark:
                    self.network.add_zero_order_edge(landmark, best_landmark)

                x_denoised = self._denoise_local(x, best_landmark)
                self._update_landmark(x, best_landmark)
                return x_denoised
            else:
                # no suitable landmark found -> create new one
                self._initialize_new_landmark(x)
                return x.copy()

    def _perform_traversal(self, x, calc_mults=True):
        """
        Perform manifold traversal to find best landmark for point x.
        
        Returns:
            landmark: Final landmark object
            phi: Final objective value  
            trajectory: List of visited landmarks
            edge_orders: List of edge orders used (0 or 1)
            mults: Number of multiplications performed
        """
        mults = 0
        if calc_mults:
            mults += self.ambient_dim  # initial objective computation

        current_landmark = self.network.landmarks[0]  # start at first landmark
        converged = False
        trajectory = [current_landmark]
        edge_orders = []

        # initial objective value
        phi = np.sum((current_landmark.position - x) ** 2)

        while not converged:
            # compute Riemannian gradient
            grad_phi = (current_landmark.tangent_basis.T @
                        (current_landmark.position - x))

            if calc_mults:
                mults += self.ambient_dim * self.intrinsic_dim

            # try first-order step
            first_order_edges = current_landmark.first_order_edges
            deg_1 = len(first_order_edges)

            if deg_1 > 0:
                # find most correlated edge embedding
                best_corr = math.inf
                next_landmark = current_landmark

                for edge in first_order_edges:
                    if edge.embedding is not None:
                        corr = np.dot(edge.embedding, grad_phi)

                        if corr < best_corr:
                            best_corr = corr
                            next_landmark = edge.target

                if calc_mults:
                    mults += self.intrinsic_dim * deg_1

                # compute objective at speculated next vertex
                next_phi = np.sum((next_landmark.position - x) ** 2)
                if calc_mults:
                    mults += self.ambient_dim

                if next_phi < phi:
                    # first-order step improves objective
                    current_landmark = next_landmark
                    phi = next_phi
                    trajectory.append(current_landmark)
                    edge_orders.append(1)
                    continue

            # first-order step failed or no first-order neighbors, try zero-order
            zero_order_edges = current_landmark.zero_order_edges
            deg_0 = len(zero_order_edges)

            if deg_0 > 0:
                best_landmark = current_landmark
                best_phi = phi

                if calc_mults:
                    mults += self.ambient_dim * deg_0

                for edge in zero_order_edges:
                    neighbor_phi = np.sum((edge.target.position - x) ** 2)
                    if neighbor_phi < best_phi:
                        best_phi = neighbor_phi
                        best_landmark = edge.target

                if best_landmark is not current_landmark:
                    # zero-order step improves objective
                    current_landmark = best_landmark
                    phi = best_phi
                    trajectory.append(current_landmark)
                    edge_orders.append(0)
                    continue

            # no improvement found - converged
            converged = True

        return current_landmark, phi, trajectory, edge_orders, mults

    def _initialize_new_landmark(self, x):
        """Initialize a new landmark at point x."""
        if self.network.num_landmarks == 0:
            # first landmark -> use random tangent space
            random_matrix = np.random.randn(self.ambient_dim, self.intrinsic_dim)
            U, s, _ = svd(random_matrix, full_matrices=False)
            U_new = U[:, :self.intrinsic_dim]
            s_new = s[:self.intrinsic_dim]
            S_new_diag = np.diag(s_new)

            # add landmark to network
            landmark_idx = self.network.add_landmark(x, U_new, S_new_diag)
            new_landmark = self.network.landmarks[landmark_idx]

            # add self-edges
            self.network.add_first_order_edge(new_landmark, new_landmark)
            self.network.add_zero_order_edge(new_landmark, new_landmark)
            # update edge embedding for self-edge (should be zero)
            new_landmark.first_order_edges[0].update_embedding(np.zeros(self.intrinsic_dim))

            return new_landmark

        else:
            # not first landmark -> find neighbors and compute tangent space
            # add landmark to network
            random_matrix = np.random.randn(self.ambient_dim, self.intrinsic_dim)
            U, s, _ = svd(random_matrix, full_matrices=False)
            U_new = U[:, :self.intrinsic_dim]
            s_new = s[:self.intrinsic_dim]
            S_new_diag = np.diag(s_new)

            landmark_idx = self.network.add_landmark(x, U_new, S_new_diag)
            new_landmark = self.network.landmarks[landmark_idx]

            # find first-order neighbors within radius
            R_sq = self.R_1st_order_nbhd ** 2
            neighbor_landmarks = []

            for existing_landmark in self.network.landmarks[:-1]:  # exclude the new landmark itself
                dist_sq = np.sum((new_landmark.position - existing_landmark.position) ** 2)
                if dist_sq <= R_sq:
                    neighbor_landmarks.append(existing_landmark)
                    # add bidirectional first-order edges
                    self.network.add_first_order_edge(new_landmark, existing_landmark)
                    self.network.add_first_order_edge(existing_landmark, new_landmark)

            # add self-edge
            self.network.add_first_order_edge(new_landmark, new_landmark)

            # update tangent space if has multiple neighbors
            if len(neighbor_landmarks) > 0:
                # form matrix H with difference vectors
                H = np.zeros((self.ambient_dim, len(neighbor_landmarks)))
                for j, neighbor_landmark in enumerate(neighbor_landmarks):
                    diff_vec = (neighbor_landmark.position - new_landmark.position)
                    H[:, j] = diff_vec / np.linalg.norm(diff_vec)

                # compute SVD and update tangent space
                U, s, _ = svd(H, full_matrices=False)
                U_new = U[:, :self.intrinsic_dim]
                s_new = s[:self.intrinsic_dim]
                S_new_diag = np.diag(s_new)

                new_landmark.update_tangent_space(U_new, S_new_diag)

            # update edge embeddings for new landmark and its neighbors
            self.network.update_edge_embeddings(new_landmark)
            for neighbor_landmark in neighbor_landmarks:
                self.network.update_edge_embeddings(neighbor_landmark)

            return new_landmark

    def _update_landmark(self, x, landmark):
        """Updates landmark's point count, position and tangent space."""
        # update point count and landmark position
        P_i = landmark.point_count
        landmark.increment_point_count()

        # update landmark as running average
        new_position = ((P_i / (P_i + 1)) * landmark.position + (1 / (P_i + 1)) * x)
        landmark.update_position(new_position)

        # update tangent space using TISVD
        landmark_idx = self.network.landmarks.index(landmark)
        U_old = landmark.tangent_basis
        S_old = landmark.singular_values
        U_new, S_new_diag = TISVD_gw(x, U_old, S_old, landmark_idx, self.intrinsic_dim)

        landmark.update_tangent_space(U_new, S_new_diag)

        # update edge embeddings
        self.network.update_edge_embeddings(landmark)

        # update edge embeddings of neighbors that point to this landmark
        for other_landmark in self.network.landmarks:
            for edge in other_landmark.first_order_edges:
                if edge.target is landmark:
                    self.network.update_edge_embeddings(other_landmark)
                    break  # found one edge pointing to this landmark, update once

    def _denoise_local(self, x, landmark):
        """Denoise point x using local model at landmark."""
        Q_i = landmark.position
        T_i = landmark.tangent_basis

        # project onto local tangent space
        x_denoised = Q_i + T_i @ (T_i.T @ (x - Q_i))
        return x_denoised

    def _compute_denoising_radius_sq(self, landmark):
        """Compute squared denoising radius for landmark."""
        if self.R_is_const:
            return self.R_denoising ** 2
        else:
            P_i = landmark.point_count
            return self.prod_coeff * (
                    self.sigma ** 2 * self.ambient_dim +
                    (self.sigma ** 2 * self.ambient_dim / (P_i ** self.exp_coeff)) +
                    self.d_parallel ** 2
            )

    def _exhaustive_search(self, x):
        """Find landmark with minimum distance to x via exhaustive search."""
        best_landmark = None
        best_phi = math.inf

        for landmark in self.network.landmarks:
            phi = np.sum((landmark.position - x) ** 2)
            if phi < best_phi:
                best_phi = phi
                best_landmark = landmark

        return best_landmark, best_phi

    def plot_training_curves(self, save_path=None):
        """Plot training error curves."""
        if len(self.results.mean_MT_error) == 0:
            print("No training results to plot.")
            return

        n_samples = len(self.results.mean_MT_error)
        sigma_sq_d = [self.sigma ** 2 * self.intrinsic_dim] * n_samples
        sigma_sq_D = [self.sigma ** 2 * self.ambient_dim] * n_samples

        plt.figure(figsize=(10, 6))
        plt.plot(self.results.mean_MT_error, color='orange', label='MT training')
        plt.plot(sigma_sq_d, color='green', label=r'$\sigma^2 d$')
        plt.plot(sigma_sq_D, color='blue', label=r'$\sigma^2 D$')
        plt.legend()
        plt.xlabel('Number of Samples')
        plt.ylabel('Mean Square Error (Train)')
        plt.title('Training Error Curve vs. Number of Samples')
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

        if save_path:
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.show()

    def perform_traversal(self, x, calc_mults=True):
        """
        Public method to perform traversal (for testing/analysis).
        
        Args:
            x: Point to find best landmark for
            calc_mults: Whether to calculate multiplication count
            
        Returns:
            Tuple of (landmark_idx, phi, trajectory, edge_orders, mults)
        """
        return self._perform_traversal(x, calc_mults=calc_mults)

    def exhaustive_search(self, x, calc_mults=True):
        """
        Public method for exhaustive search (for comparison/analysis).
        
        Args:
            x: Point to search for
            calc_mults: Whether to calculate multiplication count
            
        Returns:
            Tuple of (best_idx, best_distance, mults)
        """
        mults = 0
        if calc_mults:
            mults = self.ambient_dim * self.network.num_landmarks

        best_idx, best_phi = self._exhaustive_search(x)
        return best_idx, best_phi, mults

    def first_order_only_traversal(self, x, calc_mults=True):
        """
        Perform traversal using only first-order edges (for ablation study).
        
        Args:
            x: Point to traverse for
            calc_mults: Whether to calculate multiplications
            
        Returns:
            Tuple of (landmark, phi, trajectory, edge_orders, mults)
        """
        mults = 0
        if calc_mults:
            mults += self.ambient_dim  # initial objective computation

        current_landmark = self.network.landmarks[0]  # start at first landmark
        converged = False
        trajectory = [current_landmark]
        edge_orders = []

        # initial objective value
        phi = np.sum((current_landmark.position - x) ** 2)

        while not converged:
            # gradient
            grad_phi = (current_landmark.tangent_basis.T @
                        (current_landmark.position - x))

            if calc_mults:
                mults += self.ambient_dim * self.intrinsic_dim

            # try first-order step only
            first_order_edges = current_landmark.first_order_edges
            deg_1 = len(first_order_edges)

            if deg_1 > 0:
                # find most correlated edge embedding
                best_corr = math.inf
                next_landmark = current_landmark

                for edge in first_order_edges:
                    if edge.embedding is not None:
                        corr = np.dot(edge.embedding, grad_phi)

                        if corr < best_corr:
                            best_corr = corr
                            next_landmark = edge.target

                if calc_mults:
                    mults += self.intrinsic_dim * deg_1

                # compute objective at speculated next vertex
                next_phi = np.sum((next_landmark.position - x) ** 2)
                if calc_mults:
                    mults += self.ambient_dim

                if next_phi < phi and next_landmark is not current_landmark:
                    # first-order step improves objective
                    current_landmark = next_landmark
                    phi = next_phi
                    trajectory.append(current_landmark)
                    edge_orders.append(1)
                    continue

            # no improvement found - converged
            converged = True

        return current_landmark, phi, trajectory, edge_orders, mults

    def zero_order_only_traversal(self, x, calc_mults=True):
        """
        Perform traversal using only zero-order edges (for ablation study).
        
        Args:
            x: Point to traverse for
            calc_mults: Whether to calculate multiplications
            
        Returns:
            Tuple of (landmark, trajectory, edge_orders, mults)
        """
        mults = 0

        current_landmark = self.network.landmarks[0]  # start at first landmark
        converged = False
        trajectory = [current_landmark]
        edge_orders = []

        while not converged:
            # try zero-order step only
            # include all landmarks as potential zero-order neighbors for this analysis
            deg_0 = self.network.num_landmarks
            best_landmark = current_landmark
            best_phi = math.inf

            if calc_mults:
                mults += self.ambient_dim * deg_0

            for landmark in self.network.landmarks:
                neighbor_phi = np.sum((landmark.position - x) ** 2)
                if neighbor_phi < best_phi:
                    best_phi = neighbor_phi
                    best_landmark = landmark

            if best_landmark is not current_landmark:
                # zero-order step improves objective
                current_landmark = best_landmark
                trajectory.append(current_landmark)
                edge_orders.append(0)
                continue

            # no improvement found - converged
            converged = True

        return current_landmark, trajectory, edge_orders, mults

    def analyze_performance(self, X_test, X_natural_test, num_samples=None):
        """
        Analyze network performance using different traversal methods.
        
        This method replicates the analysis functionality from the ablation study.
        
        Args:
            X_test: Test data (noisy)
            X_natural_test: Clean test data 
            num_samples: Number of samples to analyze (None for all)
            
        Returns:
            Dictionary with performance metrics for different methods
        """
        if num_samples is None:
            num_samples = X_test.shape[1]
        else:
            num_samples = min(num_samples, X_test.shape[1])

        if self.network.num_landmarks == 0:
            raise ValueError("Network has no landmarks. Train the network first.")

        # initialize accumulators
        exh_total_mults = 0
        mt_total_mults = 0
        fom_total_mults = 0  # first-order only
        zom_total_mults = 0  # zero-order only

        exh_distances = []
        mt_distances = []
        fom_distances = []
        zom_distances = []

        for i in range(num_samples):
            x = X_test[:, i]
            x_nat = X_natural_test[:, i]

            # exhaustive search
            landmark_exh, _, exh_mults = self.exhaustive_search(x, calc_mults=True)
            exh_total_mults += exh_mults

            # mixed-order traversal
            landmark_mt, _, _, _, mt_mults = self.perform_traversal(x, calc_mults=True)
            mt_total_mults += mt_mults

            # first-order only traversal
            landmark_fom, _, _, _, fom_mults = self.first_order_only_traversal(x, calc_mults=True)
            fom_total_mults += fom_mults

            # zero-order only traversal
            landmark_zom, _, _, zom_mults = self.zero_order_only_traversal(x, calc_mults=True)
            zom_total_mults += zom_mults

            # compute squared distances to clean data
            SQdist_exh = np.sum((x_nat - landmark_exh.position) ** 2)
            SQdist_mt = np.sum((x_nat - landmark_mt.position) ** 2)
            SQdist_fom = np.sum((x_nat - landmark_fom.position) ** 2)
            SQdist_zom = np.sum((x_nat - landmark_zom.position) ** 2)

            exh_distances.append(SQdist_exh)
            mt_distances.append(SQdist_mt)
            fom_distances.append(SQdist_fom)
            zom_distances.append(SQdist_zom)

        # compute averages
        avg_exh_dist = np.mean(exh_distances)
        avg_mt_dist = np.mean(mt_distances)
        avg_fom_dist = np.mean(fom_distances)
        avg_zom_dist = np.mean(zom_distances)

        avg_exh_mults = exh_total_mults / num_samples
        avg_mt_mults = mt_total_mults / num_samples
        avg_fom_mults = fom_total_mults / num_samples
        avg_zom_mults = zom_total_mults / num_samples

        return {
            'exhaustive': {'avg_distance': avg_exh_dist, 'avg_mults': avg_exh_mults},
            'mixed_order': {'avg_distance': avg_mt_dist, 'avg_mults': avg_mt_mults},
            'first_order_only': {'avg_distance': avg_fom_dist, 'avg_mults': avg_fom_mults},
            'zero_order_only': {'avg_distance': avg_zom_dist, 'avg_mults': avg_zom_mults},
            'network_stats': self.network.get_network_stats()
        }
