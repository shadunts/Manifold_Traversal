import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from models.landmark import Landmark


class TraversalNetwork:
    """
    Encapsulates the manifold traversal network structure.
    Contains a list of Landmark objects and provides methods for network operations.
    """

    def __init__(self):
        self.landmarks = []  # list of Landmark objects

    @property
    def num_landmarks(self):
        """Number of landmarks in the network."""
        return len(self.landmarks)

    def add_landmark(self, position, tangent_basis, singular_vals, point_count=1, color=None):
        """Add a new landmark to the network."""
        landmark = Landmark(position, tangent_basis, singular_vals, point_count, color)
        self.landmarks.append(landmark)
        return len(self.landmarks) - 1  # Return index of new landmark

    def add_first_order_edge(self, from_landmark_idx, to_landmark_idx, weight=1.0):
        """Add a first-order edge between landmarks."""
        from_landmark = self.landmarks[from_landmark_idx]
        return from_landmark.add_first_order_edge(to_landmark_idx, weight)

    def add_zero_order_edge(self, from_landmark_idx, to_landmark_idx, weight=1.0):
        """Add a zero-order edge between landmarks."""
        from_landmark = self.landmarks[from_landmark_idx]
        return from_landmark.add_zero_order_edge(to_landmark_idx, weight)

    def update_edge_embeddings(self, landmark_idx):
        """Update edge embeddings for a specific landmark."""
        landmark = self.landmarks[landmark_idx]
        landmark.update_edge_embeddings(self.landmarks)

    def get_landmark_positions(self):
        """Get all landmark positions as a numpy array."""
        if not self.landmarks:
            return np.array([])
        return np.column_stack([landmark.position for landmark in self.landmarks])

    def get_network_stats(self):
        """Return summary statistics about the network."""
        if not self.landmarks:
            return {
                'num_landmarks': 0,
                'total_first_order_edges': 0,
                'total_zero_order_edges': 0,
                'total_points_assigned': 0
            }

        return {
            'num_landmarks': self.num_landmarks,
            'total_first_order_edges': sum(len(landmark.first_order_edges)
                                           for landmark in self.landmarks),
            'total_zero_order_edges': sum(len(landmark.zero_order_edges)
                                          for landmark in self.landmarks),
            'total_points_assigned': sum(landmark.point_count for landmark in self.landmarks)
        }

    def visualize(self, show_edges=True, show_tangent_spaces=False):
        """
        Visualization of the network structure.

        Args:
            show_edges: Whether to show connections between landmarks
            show_tangent_spaces: Whether to visualize tangent spaces
        """
        if self.num_landmarks == 0:
            print("No landmarks to visualize.")
            return

        landmarks_array = self.get_landmark_positions()
        point_counts = [landmark.point_count for landmark in self.landmarks]

        if landmarks_array.shape[0] == 2:
            # 2D visualization
            plt.figure(figsize=(10, 8))

            # landmarks colored by point count
            scatter = plt.scatter(landmarks_array[0, :], landmarks_array[1, :],
                                  c=point_counts, cmap='viridis', s=100, alpha=0.7)
            plt.colorbar(scatter, label='Points per Landmark')

            # edges if requested
            if show_edges:
                for i, landmark in enumerate(self.landmarks):
                    for edge in landmark.first_order_edges.values():
                        if edge.target_idx != i:  # Skip self-edges
                            target_idx = edge.target_idx
                            plt.plot([landmarks_array[0, i], landmarks_array[0, target_idx]],
                                     [landmarks_array[1, i], landmarks_array[1, target_idx]],
                                     'b-', alpha=0.3, linewidth=1)

            plt.title(f'Network Visualization ({self.num_landmarks} landmarks)')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.show()

        elif landmarks_array.shape[0] >= 3:
            # 3D visualization (use first 3 dimensions for high-dimensional data)
            fig = plt.figure(figsize=(12, 5))

            # 3D network structure
            ax1 = fig.add_subplot(121, projection='3d')

            # plot landmarks colored by point count
            scatter = ax1.scatter(landmarks_array[0, :], landmarks_array[1, :], landmarks_array[2, :],
                                  c=point_counts, cmap='viridis', s=80, alpha=0.8)
            plt.colorbar(scatter, ax=ax1, label='Points per Landmark')

            if show_edges:
                # draw first-order edges (blue)
                for i, landmark in enumerate(self.landmarks):
                    for edge in landmark.first_order_edges.values():
                        if edge.target_idx != i:  # skip self-edges
                            target_idx = edge.target_idx
                            ax1.plot([landmarks_array[0, i], landmarks_array[0, target_idx]],
                                     [landmarks_array[1, i], landmarks_array[1, target_idx]],
                                     [landmarks_array[2, i], landmarks_array[2, target_idx]],
                                     'b-', alpha=0.3, linewidth=1)

                # draw zero-order edges (red)
                for i, landmark in enumerate(self.landmarks):
                    for edge in landmark.zero_order_edges:
                        if edge.target_idx != i:
                            target_idx = edge.target_idx
                            # only draw if not already connected by first-order edge
                            is_first_order = any(
                                fo_edge.target_idx == edge.target_idx for fo_edge in
                                landmark.first_order_edges.values())
                            if not is_first_order:
                                ax1.plot([landmarks_array[0, i], landmarks_array[0, target_idx]],
                                         [landmarks_array[1, i], landmarks_array[1, target_idx]],
                                         [landmarks_array[2, i], landmarks_array[2, target_idx]],
                                         'r-', alpha=0.2, linewidth=0.5)

            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_title('3D Network Structure\n(Blue=1st order, Red=0th order)')

            # 2D projection
            ax2 = fig.add_subplot(122)
            scatter2 = ax2.scatter(landmarks_array[0, :], landmarks_array[1, :],
                                   c=point_counts, cmap='viridis', s=60, alpha=0.7)
            plt.colorbar(scatter2, ax=ax2, label='Points per Landmark')

            if show_edges:
                # draw first-order edges
                for i, landmark in enumerate(self.landmarks):
                    for edge in landmark.first_order_edges.values():
                        if edge.target_idx != i:
                            target_idx = edge.target_idx
                            ax2.plot([landmarks_array[0, i], landmarks_array[0, target_idx]],
                                     [landmarks_array[1, i], landmarks_array[1, target_idx]],
                                     'b-', alpha=0.3, linewidth=1)

            ax2.set_xlabel('First Dimension')
            ax2.set_ylabel('Second Dimension')
            ax2.set_title('2D Network Projection')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        else:
            print(f"Visualization not supported for {landmarks_array.shape[0]}D data.")

    def visualize_gravitational_waves(self, gw_data=None, data_path=None, azim=45, elev=30, alpha=0.6, show_data=True):
        visualizer = GravitationalWaveVisualizer()
        if gw_data is not None:
            visualizer.set_data(gw_data)
        elif data_path is not None:
            visualizer.load_data(data_path)

        visualizer.visualize_with_network(self, azim=azim, elev=elev, alpha=alpha, show_data=show_data)


class GravitationalWaveVisualizer:
    def __init__(self, gw_params=None):
        self.data = None
        self.pca_result = None
        self.centering_mean = None
        self.shift = 0.7
        self.gw_params = {
            'PCA_U': np.eye(2),  # U matrix from PCA
            'data_path': None
        } if gw_params is None else gw_params

    def load_data(self, data_path):
        self.gw_params['data_path'] = data_path
        temp_data = np.load(data_path)
        self.data = temp_data.T
        self._compute_pca(temp_data)
        return self.data

    def set_data(self, data):
        self.data = data.T if data.shape[0] < data.shape[1] else data
        self._compute_pca(data)
        return self.data

    def generate_clean_samples(self, N):
        temp_data = np.load(self.gw_params['data_path'])
        self.data = temp_data.T
        self._compute_pca(temp_data)
        D = self.data.shape[0]  # ambient dim
        return self.data[:, :N], D

    def _compute_pca(self, data, n_components=3):
        pca = PCA(n_components=n_components)
        self.pca_result = pca.fit_transform(data)
        self.centering_mean = pca.mean_
        self.gw_params['PCA_U'] = pca.components_.T

    def plot_ground_truth_manifold(self, ax, azim=45, elev=30, alpha=0.6):
        center_shift = self.gw_params['PCA_U'].T @ self.centering_mean

        ax.scatter(self.pca_result[:, 0] + center_shift[0],
                   self.pca_result[:, 1] + center_shift[1],
                   self.pca_result[:, 2] + center_shift[2],
                   c='orange',
                   alpha=alpha,
                   s=10)

        ax.view_init(elev=elev, azim=azim)

    def visualization_embedding(self, q):
        P_vis = self.gw_params['PCA_U']
        q_centered = q - self.centering_mean

        # project: (q - mean) @ components.T
        q_vis = np.dot(q_centered, P_vis)
        return q_vis, P_vis

    def center_axes(self, ax):
        s = self.shift
        ax.set_xlim([-s, s])
        ax.set_ylim([-s, s])
        ax.set_zlim([-s, s])

    def visualize_with_network(self, network, azim=45, elev=30, alpha=0.6, show_landmarks=True, single_plot=True,
                               show_data=True):
        if single_plot:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            if show_data:
                self.plot_ground_truth_manifold(ax, azim=azim, elev=elev, alpha=alpha)

        if show_landmarks and network.num_landmarks > 0:
            landmarks_array = network.get_landmark_positions()

            vis_landmarks = []
            for i in range(landmarks_array.shape[1]):
                vis_point, _ = self.visualization_embedding(landmarks_array[:, i])
                vis_landmarks.append(vis_point)
            vis_landmarks = np.array(vis_landmarks).T

            center_shift = self.gw_params['PCA_U'].T @ self.centering_mean

            # landmarks
            ax.scatter(vis_landmarks[0, :] + center_shift[0],
                       vis_landmarks[1, :] + center_shift[1],
                       vis_landmarks[2, :] + center_shift[2],
                       c='blue', s=2, alpha=0.6, marker='o', zorder=10)

            # first-order edges
            for i, landmark in enumerate(network.landmarks):
                for edge in landmark.first_order_edges.values():
                    if edge.target_idx != i:
                        j = edge.target_idx
                        ax.plot([vis_landmarks[0, i] + center_shift[0], vis_landmarks[0, j] + center_shift[0]],
                                [vis_landmarks[1, i] + center_shift[1], vis_landmarks[1, j] + center_shift[1]],
                                [vis_landmarks[2, i] + center_shift[2], vis_landmarks[2, j] + center_shift[2]],
                                'b-', alpha=0.6, linewidth=1, zorder=5)

            # zero-order edges
            for i, landmark in enumerate(network.landmarks):
                for edge in landmark.zero_order_edges:
                    if edge.target_idx != i:
                        j = edge.target_idx
                        ax.plot([vis_landmarks[0, i] + center_shift[0], vis_landmarks[0, j] + center_shift[0]],
                                [vis_landmarks[1, i] + center_shift[1], vis_landmarks[1, j] + center_shift[1]],
                                [vis_landmarks[2, i] + center_shift[2], vis_landmarks[2, j] + center_shift[2]],
                                'r-', alpha=0.6, linewidth=1.5, zorder=5)

        self.center_axes(ax)
        ax.view_init(elev=elev, azim=azim)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
        ax.grid(False)
        ax.set_axis_off()

        plt.tight_layout()
        plt.show()