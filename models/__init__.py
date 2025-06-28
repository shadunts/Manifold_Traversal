"""
Manifold Traversal Models Package

This package contains the core classes for manifold traversal:
- Edge: Base class and specific edge types (FirstOrderEdge, ZeroOrderEdge)
- Landmark: Individual landmark representation
- TraversalNetwork: Network of connected landmarks
- ManifoldTraversal: Main algorithm implementation
- TrainingResults: Training metrics and results storage
"""

from .edge import Edge, FirstOrderEdge, ZeroOrderEdge
from .landmark import Landmark
from .traversal_network import TraversalNetwork
from .manifold_traversal import ManifoldTraversal
from .training_results import TrainingResults

__all__ = [
    'Edge', 'FirstOrderEdge', 'ZeroOrderEdge',
    'Landmark', 
    'TraversalNetwork',
    'ManifoldTraversal',
    'TrainingResults'
] 