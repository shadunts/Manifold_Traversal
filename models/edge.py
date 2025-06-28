class Edge:
    """
    Base class for edges between landmarks in the manifold traversal network.
    Encapsulates common edge properties: target landmark and weight.
    """

    def __init__(self, target_landmark, weight=1.0):
        """
        Initialize an edge.

        Args:
            target_landmark: Target Landmark object
            weight: Edge weight
        """
        self.target = target_landmark
        self.weight = weight

    def __repr__(self):
        return f"{self.__class__.__name__}(target=Landmark, weight={self.weight})"

class FirstOrderEdge(Edge):
    """
    First-order edge with embedding in tangent space.
    Used for gradient-based traversal steps.
    """

    def __init__(self, target_landmark, weight=1.0, embedding=None):
        super().__init__(target_landmark, weight)
        self.embedding = embedding

    def update_embedding(self, embedding):
        """Update the edge embedding."""
        self.embedding = embedding.copy() if embedding is not None else None

    def __repr__(self):
        return f"FirstOrderEdge(target=Landmark, weight={self.weight}, has_embedding={self.embedding is not None})"


class ZeroOrderEdge(Edge):
    """
    Zero-order edge without embedding.
    Used for exhaustive search fallback steps.
    """

    def __init__(self, target_landmark, weight=1.0):
        super().__init__(target_landmark, weight)

    def __repr__(self):
        return f"ZeroOrderEdge(target=Landmark, weight={self.weight})"
