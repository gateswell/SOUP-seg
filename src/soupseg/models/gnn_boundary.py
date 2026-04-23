"""
GNN-based Boundary Detection for SoupSeg v1.1.0.

PyTorch implementation of a Graph Attention Network (GAT) for cell boundary
detection using the cell graph representation. Each cell is a node, edges
connect neighboring cells, and node/edge features encode boundary likelihood.

This approach is complementary to U-Net:
- U-Net: pixel-level boundary heatmap from image
- GNN: cell-level boundary refinement using graph structure

Architecture:
- Node features: cell area, eccentricity, mean intensity, centroid position
- Edge features: distance between centroids, shared boundary length
- Model: 2-layer Graph Attention Network (GAT)
- Output: boundary probability for each edge (cell-cell interface)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not found. GNN boundary detection requires: pip install torch"
    )


# ---------------------------------------------------------------------------
# Graph construction utilities
# ---------------------------------------------------------------------------

def build_cell_graph(
    cell_properties: List[Dict],
    cell_mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    max_neighbors: int = 10,
    distance_threshold_um: float = 50.0,
    pixel_size_um: float = 0.5,
) -> Dict[str, Any]:
    """
    Build a cell graph from detected cells.

    Each cell becomes a node. Edges connect cells whose boundaries are adjacent
    or whose centroids are within distance_threshold_um.

    Args:
        cell_properties: List of cell property dicts from nuclei.py.
        cell_mask: Labeled cell segmentation mask (H, W).
        image: Optional ssDNA image for intensity features.
        max_neighbors: Maximum number of neighbor edges per node.
        distance_threshold_um: Maximum centroid distance for edge creation.
        pixel_size_um: Pixel size in microns.

    Returns:
        Dict with:
        - node_features: (N, F) tensor of node features
        - edge_index: (2, E) tensor of edge connectivity
        - edge_features: (E, G) tensor of edge features
        - node_to_cell: Dict mapping node_idx -> cell_id
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GNN boundary detection.")

    n_cells = len(cell_properties)
    if n_cells == 0:
        return {
            "node_features": torch.empty(0, 0),
            "edge_index": torch.empty(2, 0, dtype=torch.long),
            "edge_features": torch.empty(0, 0),
            "node_to_cell": {},
        }

    device = torch.device("cpu")

    # --- Node features ---
    # Features: [area_norm, eccentricity, intensity_mean, intensity_std,
    #            centroid_x_norm, centroid_y_norm]
    node_feats = []

    # Normalization factors
    all_areas = [p.get("area_um2", 0) for p in cell_properties]
    max_area = max(all_areas) if all_areas else 1.0

    h, w = cell_mask.shape
    img_mean = image.mean() if image is not None else 0.5
    img_std = image.std() if image is not None else 0.5

    for i, prop in enumerate(cell_properties):
        area_norm = prop.get("area_um2", 0) / max_area
        ecc = prop.get("eccentricity", 0.5)
        cy, cx = prop.get("centroid", (h / 2, w / 2))

        # Intensity features
        if image is not None and prop.get("label", 0) > 0:
            cell_binary = (cell_mask == prop["label"])
            intensities = image[cell_binary]
            if len(intensities) > 0:
                int_mean = (intensities.mean() - img_mean) / (img_std + 1e-8)
                int_std = intensities.std() / (img_std + 1e-8)
            else:
                int_mean = 0.0
                int_std = 0.0
        else:
            int_mean = 0.0
            int_std = 0.0

        cx_norm = cx / w
        cy_norm = cy / h

        node_feats.append([
            area_norm, ecc, int_mean, int_std, cx_norm, cy_norm
        ])

    node_features = torch.tensor(node_feats, dtype=torch.float32, device=device)

    # --- Build edges ---
    from scipy.spatial import KDTree

    centroids = np.array([
        (p.get("centroid", (h / 2, w / 2))[1],  # x
         p.get("centroid", (h / 2, w / 2))[0])  # y
        for p in cell_properties
    ])  # (N, 2) in (x, y) order

    # KDTree for efficient neighbor search
    tree = KDTree(centroids)
    edge_list = []
    edge_feats = []

    threshold_px = distance_threshold_um / pixel_size_um

    for i in range(n_cells):
        # Find neighbors within threshold
        distances, indices = tree.query(centroids[i], k=min(max_neighbors + 1, n_cells))
        for dist, j in zip(distances, indices):
            if j <= i:
                continue
            if dist > threshold_px:
                continue
            edge_list.append((i, j))
            # Edge features: [distance_norm, area_sum_norm, area_diff_norm]
            dist_norm = dist / threshold_px
            area_i = cell_properties[i].get("area_um2", 0) / max_area
            area_j = cell_properties[j].get("area_um2", 0) / max_area
            area_sum = (area_i + area_j)
            area_diff = abs(area_i - area_j)
            edge_feats.append([dist_norm, area_sum, area_diff])

    if len(edge_list) == 0:
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_features = torch.empty(0, 3, dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
        edge_features = torch.tensor(edge_feats, dtype=torch.float32, device=device)

    node_to_cell = {i: cell_properties[i].get("cell_id", f"cell_{i:05d}")
                    for i in range(n_cells)}

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "node_to_cell": node_to_cell,
        "n_nodes": n_cells,
        "n_edges": len(edge_list),
    }


# ---------------------------------------------------------------------------
# GAT Model
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class GATConv(nn.Module):
        """Graph Attention Convolution layer."""

        def __init__(self, in_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.1):
            super().__init__()
            self.heads = heads
            self.out_channels = out_channels
            self.sqrt_h = (out_channels // heads) ** 0.5

            self.W = nn.Linear(in_channels, out_channels, bias=False)
            self.att = nn.Parameter(torch.empty(1, heads, out_channels // heads))
            self.dropout = nn.Dropout(dropout)

            nn.init.xavier_uniform_(self.att)

        def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
            """
            Args:
                x: Node features (N, F)
                edge_index: Edge connectivity (2, E)

            Returns:
                Updated node features (N, out_channels)
            """
            N = x.size(0)

            # Linear transform
            h = self.W(x)  # (N, out_channels)
            h = h.view(N, self.heads, -1)  # (N, heads, F')

            # Self-attention
            i, j = edge_index  # (E,)

            # Compute attention scores
            h_i = h[i]  # (E, heads, F')
            h_j = h[j]  # (E, heads, F')
            att_score = (h_i * h_j).sum(dim=-1) / self.sqrt_h  # (E, heads)

            # Softmax over neighbors
            att = F.softmax(att_score, dim=0)  # (E, heads)
            att = self.dropout(att)

            # Aggregate
            out = torch.zeros(N, self.heads, h.size(-1), device=x.device)
            out.index_add_(0, i, att * h_j)  # scatter_add

            return out.view(N, -1)


    class GNNBoundaryDetector(nn.Module):
        """
        Graph Attention Network for cell boundary detection.

        Input: Cell graph with node and edge features
        Output: Boundary probability for each edge (cell-cell interface)

        The model learns to predict which cell-cell interfaces correspond to
        real cell boundaries vs. artifacts from over-segmentation.
        """

        def __init__(
            self,
            node_features: int = 6,
            edge_features: int = 3,
            hidden_dim: int = 64,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.node_features = node_features
            self.edge_features = edge_features
            self.hidden_dim = hidden_dim

            # Node embedding
            self.node_embed = nn.Linear(node_features, hidden_dim)

            # Edge embedding
            self.edge_embed = nn.Linear(edge_features, hidden_dim)

            # GAT layers
            self.gat_layers = nn.ModuleList()
            for _ in range(num_layers):
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout)
                )

            # Edge prediction head
            # Takes concatenated source node, target node, and edge features
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            node_features: Tensor,
            edge_index: Tensor,
            edge_features: Tensor,
        ) -> Tensor:
            """
            Forward pass.

            Args:
                node_features: (N, node_features)
                edge_index: (2, E)
                edge_features: (E, edge_features)

            Returns:
                Boundary probabilities (E,) for each edge.
            """
            N = node_features.size(0)
            if N == 0 or edge_index.size(1) == 0:
                return torch.empty(0, device=node_features.device)

            # Embed
            h = self.node_embed(node_features)  # (N, hidden)
            e = self.edge_embed(edge_features)  # (E, hidden)

            # GAT layers
            for gat in self.gat_layers:
                h = gat(h, edge_index)  # (N, hidden)
                h = F.elu(h)

            # Edge prediction
            i, j = edge_index  # (E,)
            h_i = h[i]  # (E, hidden)
            h_j = h[j]  # (E, hidden)

            # Concatenate: [h_src, h_dst, edge_embedding]
            edge_input = torch.cat([h_i, h_j, e], dim=-1)  # (E, hidden*3)

            boundary_prob = self.edge_mlp(edge_input).squeeze(-1)  # (E,)

            return boundary_prob


# ---------------------------------------------------------------------------
# GNN Boundary Refinement
# ---------------------------------------------------------------------------

@dataclass
class GNNConfig:
    """Configuration for GNN boundary detector."""
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    distance_threshold_um: float = 50.0
    max_neighbors: int = 10
    boundary_threshold: float = 0.5
    device: str = "auto"

    def get_device(self) -> str:
        if self.device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return self.device


def create_gnn_model(config: Optional[GNNConfig] = None) -> "GNNBoundaryDetector":
    """Create a GNN boundary detector model."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GNN boundary detection.")

    if config is None:
        config = GNNConfig()

    model = GNNBoundaryDetector(
        node_features=6,
        edge_features=3,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )

    device = config.get_device()
    model.to(device)

    return model


def predict_boundary_edges(
    cell_properties: List[Dict],
    cell_mask: np.ndarray,
    image: Optional[np.ndarray],
    model: Optional["GNNBoundaryDetector"] = None,
    config: Optional[GNNConfig] = None,
) -> Dict[int, float]:
    """
    Predict boundary probabilities for cell-cell edges.

    Args:
        cell_properties: List of cell property dicts.
        cell_mask: Labeled cell mask.
        image: Optional ssDNA image.
        model: Pre-trained GNN model. Created from config if None.
        config: Model configuration.

    Returns:
        Dict mapping (i, j) edge tuple -> boundary probability.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GNN boundary detection.")

    if config is None:
        config = GNNConfig()

    device = config.get_device()

    # Build graph
    graph = build_cell_graph(
        cell_properties=cell_properties,
        cell_mask=cell_mask,
        image=image,
        max_neighbors=config.max_neighbors,
        distance_threshold_um=config.distance_threshold_um,
    )

    if graph["n_edges"] == 0:
        return {}

    if model is None:
        model = create_gnn_model(config)

    model.eval()
    model.to(device)

    with torch.no_grad():
        node_features = graph["node_features"].to(device)
        edge_index = graph["edge_index"].to(device)
        edge_features = graph["edge_features"].to(device)

        probs = model(node_features, edge_index, edge_features)
        probs = probs.cpu().numpy()

    # Map edge index to probabilities
    edge_probs = {}
    edge_index_np = graph["edge_index"].numpy()
    for k in range(edge_index_np.shape[1]):
        i = int(edge_index_np[0, k])
        j = int(edge_index_np[1, k])
        edge_probs[(i, j)] = float(probs[k])

    return edge_probs


def refine_boundaries_by_gnn(
    cell_mask: np.ndarray,
    cell_properties: List[Dict],
    image: np.ndarray,
    edge_probs: Dict[int, float],
    boundary_threshold: float = 0.5,
    merge: bool = True,
) -> np.ndarray:
    """
    Refine cell segmentation using GNN boundary predictions.

    For each edge with low boundary probability (below threshold),
    consider merging the two cells.

    Args:
        cell_mask: Current cell segmentation mask (H, W).
        cell_properties: List of cell properties.
        image: ssDNA image for reference.
        edge_probs: Dict mapping (i, j) -> boundary probability.
        boundary_threshold: If edge prob < threshold, merge cells.
        merge: Whether to actually merge cells (False = just flag edges).

    Returns:
        Refined cell mask (same shape as input).
    """
    if not merge:
        # Just return original mask with annotations
        return cell_mask

    # Build label->idx mapping
    label_to_idx = {}
    for idx, prop in enumerate(cell_properties):
        label_to_idx[prop["label"]] = idx

    # Identify edges to merge
    edges_to_merge = []
    for (i, j), prob in edge_probs.items():
        if prob < boundary_threshold:
            # Get cell labels for this edge
            label_i = None
            label_j = None
            for idx, prop in enumerate(cell_properties):
                if idx == i:
                    label_i = prop["label"]
                elif idx == j:
                    label_j = prop["label"]

            if label_i is not None and label_j is not None:
                edges_to_merge.append((label_i, label_j))

    if not edges_to_merge:
        return cell_mask

    # Apply merges: replace label_j with label_i wherever they overlap
    refined = cell_mask.copy()

    for label_i, label_j in edges_to_merge:
        # Find boundary region
        mask_i = (refined == label_i)
        mask_j = (refined == label_j)

        # Create dilated version of each to find overlap region
        from scipy.ndimage import binary_dilation, binary_erosion
        struct = np.ones((5, 5), dtype=bool)

        boundary_i = binary_dilation(mask_i, structure=struct) & ~mask_i & mask_j
        boundary_j = binary_dilation(mask_j, structure=struct) & ~mask_j & mask_i

        # Merge: assign boundary pixels to larger cell
        area_i = mask_i.sum()
        area_j = mask_j.sum()

        if area_i >= area_j:
            # Merge j into i
            refined[boundary_j] = label_i
            # Also reassign interior of j if small
            if area_j < area_i * 0.1:  # j is <10% size of i
                refined[mask_j] = label_i
        else:
            refined[boundary_i] = label_j
            if area_i < area_j * 0.1:
                refined[mask_i] = label_j

    return refined


# ---------------------------------------------------------------------------
# Train GNN (simplified)
# ---------------------------------------------------------------------------

def train_gnn(
    train_graphs: List[Dict],
    val_graphs: Optional[List[Dict]] = None,
    config: Optional[GNNConfig] = None,
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    edge_labels: Optional[List[np.ndarray]] = None,
    val_labels: Optional[List[np.ndarray]] = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train the GNN boundary detector.

    Args:
        train_graphs: List of graph dicts from build_cell_graph.
        val_graphs: Optional validation graphs.
        config: Model configuration.
        epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        edge_labels: List of binary arrays (E,) indicating true boundaries.
        val_labels: Validation edge labels.
        save_path: Path to save best model weights.
        verbose: Print progress.

    Returns:
        Training history dict.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GNN training.")

    if config is None:
        config = GNNConfig()

    device = config.get_device()
    model = create_gnn_model(config)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')

    n_train = len(train_graphs)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        indices = np.random.permutation(n_train)

        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_loss = 0.0

            for j, idx in enumerate(batch_idx):
                graph = train_graphs[idx]
                labels = edge_labels[idx] if edge_labels is not None else None

                if graph["n_edges"] == 0 or labels is None:
                    continue

                node_feat = graph["node_features"].to(device)
                edge_idx = graph["edge_index"].to(device)
                edge_feat = graph["edge_features"].to(device)
                target = torch.from_numpy(labels).float().to(device)

                pred = model(node_feat, edge_idx, edge_feat)

                loss = criterion(pred, target)
                batch_loss += loss

            if batch_loss > 0:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
                n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # Validation
        val_loss_val = 0.0
        if val_graphs is not None and val_labels is not None:
            model.eval()
            n_val_batches = 0
            val_total_loss = 0.0

            with torch.no_grad():
                for g, l in zip(val_graphs, val_labels):
                    if g["n_edges"] == 0:
                        continue
                    node_feat = g["node_features"].to(device)
                    edge_idx = g["edge_index"].to(device)
                    edge_feat = g["edge_features"].to(device)
                    target = torch.from_numpy(l).float().to(device)

                    pred = model(node_feat, edge_idx, edge_feat)
                    loss = criterion(pred, target)
                    val_total_loss += loss.item()
                    n_val_batches += 1

            if n_val_batches > 0:
                val_loss_val = val_total_loss / n_val_batches
                history["val_loss"].append(val_loss_val)

                if save_path and val_loss_val < best_val_loss:
                    best_val_loss = val_loss_val
                    torch.save(model.state_dict(), save_path)
        else:
            if save_path and avg_loss < best_val_loss:
                best_val_loss = avg_loss
                torch.save(model.state_dict(), save_path)

        if verbose and (epoch + 1) % 10 == 0:
            msg = f"Epoch {epoch+1}/{epochs} - train_loss: {avg_loss:.4f}"
            if val_graphs is not None:
                msg += f" - val_loss: {val_loss_val:.4f}"
            print(msg)

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "model": model,
    }
