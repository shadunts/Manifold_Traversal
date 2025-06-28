import numpy as np
import matplotlib.pyplot as plt

from .landmark import Landmark

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

    def add_first_order_edge(self, from_landmark, to_landmark, weight=1.0):
        """Add a first-order edge between landmarks."""
        return from_landmark.add_first_order_edge(to_landmark, weight)

    def add_zero_order_edge(self, from_landmark, to_landmark, weight=1.0):
        """Add a zero-order edge between landmarks."""
        return from_landmark.add_zero_order_edge(to_landmark, weight)

    def update_edge_embeddings(self, landmark):
        """Update edge embeddings for a specific landmark."""
        landmark.update_edge_embeddings()

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

        if landmarks_array.shape[0] == 2:
            # 2D visualization
            plt.figure(figsize=(10, 8))

            # landmarks
            plt.scatter(landmarks_array[0, :], landmarks_array[1, :],
                        c='red', s=100, alpha=0.7, label='Landmarks')

            # edges if requested
            if show_edges:
                for i, landmark in enumerate(self.landmarks):
                    for edge in landmark.first_order_edges:
                        if edge.target is not landmark:  # Skip self-edges
                            # Find target landmark index for plotting
                            target_idx = self.landmarks.index(edge.target)
                            plt.plot([landmarks_array[0, i], landmarks_array[0, target_idx]],
                                     [landmarks_array[1, i], landmarks_array[1, target_idx]],
                                     'b-', alpha=0.3, linewidth=1)

            plt.title(f'Network Visualization ({self.num_landmarks} landmarks)')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.show()

        elif landmarks_array.shape[0] == 3:
            # 3D visualization
            import plotly.graph_objects as go

            fig = go.Figure()

            # landmarks
            fig.add_trace(go.Scatter3d(
                x=landmarks_array[0, :],
                y=landmarks_array[1, :],
                z=landmarks_array[2, :],
                mode='markers',
                marker=dict(size=8, color='red', opacity=0.7, showscale=False),
                name='Landmarks'
            ))

            # edges if requested
            if show_edges:
                for i, landmark in enumerate(self.landmarks):
                    for edge in landmark.first_order_edges:
                        if edge.target is not landmark:  # skip self-edges
                            # find target landmark index for plotting
                            target_idx = self.landmarks.index(edge.target)
                            fig.add_trace(go.Scatter3d(
                                x=[landmarks_array[0, i], landmarks_array[0, target_idx]],
                                y=[landmarks_array[1, i], landmarks_array[1, target_idx]],
                                z=[landmarks_array[2, i], landmarks_array[2, target_idx]],
                                mode='lines',
                                line=dict(color='blue', width=2),
                                opacity=0.3,
                                showlegend=False
                            ))

            fig.update_layout(
                title=f'Network Visualization ({self.num_landmarks} landmarks)',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data',
                    aspectratio=dict(x=1, y=1, z=1)
                ),
                width=800,
                height=800
            )
            fig.show()

        else:
            print(f"Visualization not supported for {landmarks_array.shape[0]}D data. "
                  f"Use specialized visualization functions for high-dimensional data.")