import numpy as np
import matplotlib.pyplot as plt

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
                                fo_edge.target_idx == edge.target_idx for fo_edge in landmark.first_order_edges.values())
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
