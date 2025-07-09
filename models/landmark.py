import numpy as np
import matplotlib.colors as mcolors
from models.edge import FirstOrderEdge, ZeroOrderEdge

# get all CSS4 colors for visualization
all_colors = list(mcolors.CSS4_COLORS)


class Landmark:
    """
    Represents a single landmark in the manifold traversal network.
    Encapsulates all properties and neighbors of a landmark point.
    """

    def __init__(self, position, tangent_basis, singular_vals, point_count=1, color=None):
        """
        Initialize a landmark.

        Args:
            position: The landmark point (numpy array)
            tangent_basis: Tangent space basis matrix
            singular_vals: Singular values matrix
            point_count: Number of data points assigned to this landmark
            color: Visualization color
        """
        # core landmark properties
        self.position = position.copy()
        self.tangent_basis = tangent_basis.copy()
        self.singular_values = singular_vals.copy()
        self.point_count = point_count

        # graph connectivity
        self.first_order_edges = {}  # dict: target_idx -> Edge object
        self.zero_order_edges = []  # list of zero-order Edge objects

        # visual
        if color is None:
            color = all_colors[np.random.randint(0, len(all_colors))]
        self.color = color

    def add_first_order_edge(self, target_idx, weight=1.0):
        """Add a first-order edge to another landmark by index."""
        edge = FirstOrderEdge(target_idx, weight)
        self.first_order_edges[target_idx] = edge
        return edge

    def add_zero_order_edge(self, target_idx, weight=1.0):
        """Add a zero-order edge to another landmark by index."""
        edge = ZeroOrderEdge(target_idx, weight)
        self.zero_order_edges.append(edge)
        return edge

    def get_first_order_edge_to(self, target_idx):
        """Get the first-order edge to a specific target landmark (O(1) lookup)."""
        return self.first_order_edges.get(target_idx, None)

    def update_edge_embeddings(self, landmarks):
        """Update edge embeddings for all first-order edges."""
        for edge in self.first_order_edges.values():
            target_landmark = landmarks[edge.target_idx]
            embedding = self.tangent_basis.T @ (target_landmark.position - self.position)
            edge.update_embedding(embedding)

    def update_position(self, new_position):
        """Update the landmark position (running average during training)."""
        self.position = new_position.copy()

    def update_tangent_space(self, new_tangent_basis, new_singular_vals):
        """Update the tangent space basis and singular values."""
        self.tangent_basis = new_tangent_basis.copy()
        self.singular_values = new_singular_vals.copy()

    def increment_point_count(self):
        """Increment the count of points assigned to this landmark."""
        self.point_count += 1