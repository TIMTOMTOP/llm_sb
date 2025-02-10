import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# PyTorch Geometric modules.
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# For reproducibility.
torch.manual_seed(42)
np.random.seed(42)

###########################################
# 1. SELECT DEVICE (MPS IF AVAILABLE)     #
###########################################

# You can use mps if it's available, otherwise fall back to CPU.
# (Alternatively, you can do "cuda" if torch.cuda.is_available() for an NVIDIA GPU.)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = 'cpu' ## OVERRIDE FOR NOW, to little data for MPS(Maybe I donät knwo)
print("Using device:", device)


###########################################
# 2. LOAD THE CODE GRAPH AND PREPARE DATA #
###########################################

# Read your graph from a GraphML file.
graph = nx.read_graphml("code_graphs/sdk_graph_anthropic.graphml")

# Get nodes (with attributes) and edges (with attributes).
nodes = list(graph.nodes(data=True))
edges = list(graph.edges(data=True))

# Build a new NetworkX graph (to be sure all attributes are preserved).
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Create a fixed ordering for nodes (needed for the PyG Data object).
node_list = list(G.nodes())
node_to_idx = {node: i for i, node in enumerate(node_list)}

# Build the edge index list.
edge_list = []
edge_attr_list = []  # will hold the relationship type for each edge.
for u, v, data_ in G.edges(data=True):
    # Convert the original node names to indices.
    u_idx = node_to_idx[u]
    v_idx = node_to_idx[v]
    # Because the graph is undirected, add both (u,v) and (v,u).
    edge_list.append((u_idx, v_idx))
    edge_list.append((v_idx, u_idx))
    # Duplicate the edge attribute for the reverse edge.
    edge_attr_list.append(data_['relationship'])
    edge_attr_list.append(data_['relationship'])

# Convert the edge list into a tensor (shape [2, num_edges]).
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

#############################################
# 3. NODE INITIALIZATION BASED ON NODE TYPE #
#############################################

# Get the unique node types.
unique_node_types = sorted({ data_['type'] for _, data_ in G.nodes(data=True) })
print("Unique node types:", unique_node_types)

# Create a mapping from node type to index.
node_type_to_idx = {typ: i for i, typ in enumerate(unique_node_types)}

# For every node (in the fixed ordering), convert its type into an index.
node_type_indices = []
for node in node_list:
    typ = G.nodes[node]['type']
    node_type_indices.append(node_type_to_idx[typ])
node_type_indices = torch.tensor(node_type_indices, dtype=torch.long)

# Define the desired node embedding dimension.
node_emb_dim = 16

# Create an embedding layer that will (learn to) represent each node type.
node_type_embedding = nn.Embedding(num_embeddings=len(unique_node_types),
                                   embedding_dim=node_emb_dim)

# Now initialize the node feature matrix by "looking up" each node's type embedding.
# (Shape: [num_nodes, node_emb_dim])
x = node_type_embedding(node_type_indices)

###############################################
# 4. EDGE INITIALIZATION BASED ON RELATIONSHIP #
###############################################

# Get the unique edge relationships.
unique_edge_relationships = sorted(set(edge_attr_list))
print("Unique edge relationships:", unique_edge_relationships)

# Create a mapping from relationship to index.
edge_type_to_idx = {rel: i for i, rel in enumerate(unique_edge_relationships)}

# Convert the list of edge relationship strings to indices.
edge_type_indices = [edge_type_to_idx[rel] for rel in edge_attr_list]
edge_type_indices = torch.tensor(edge_type_indices, dtype=torch.long)

# For simplicity we set the edge embedding dimension to be the same as node_emb_dim.
edge_emb_dim = node_emb_dim

# Create an embedding layer for edge types.
edge_type_embedding = nn.Embedding(num_embeddings=len(unique_edge_relationships),
                                   embedding_dim=edge_emb_dim)

# Now initialize the edge attribute tensor.
# (Shape: [num_edges, edge_emb_dim])
edge_attr = edge_type_embedding(edge_type_indices)

###################################################
# 5. CREATE THE PYG DATA OBJECT WITH ALL FEATURES #
###################################################

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
print(data)

##########################################
# 6. DEFINE THE GNN MODEL & TRAINING LOOP #
##########################################

class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        A simple two-layer GCN.
        """
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr=None):
        # Here edge_attr is not used. To incorporate it,
        # consider using a custom MessagePassing layer.
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Instantiate the model.
model = SimpleGCN(in_channels=node_emb_dim, hidden_channels=32, out_channels=2)

####################################
# MOVE EVERYTHING TO THE MPS DEVICE #
####################################

# Move data tensors to device
data = data.to(device)

# Move the model to device
model = model.to(device)

###########################################
# 7. DEFINE CONTRASTIVE SAMPLING & LOSS   #
###########################################

def sample_contrastive_pairs(edge_index_, num_nodes, num_negatives=3):
    """
    Optimized version that minimizes CPU-GPU transfers
    """
    # Do this once at the start of training instead of every iteration
    if not hasattr(sample_contrastive_pairs, 'neighbors_dict'):
        edge_index_cpu = edge_index_.cpu().numpy()
        neighbors = {i: set() for i in range(num_nodes)}
        for src, dst in zip(edge_index_cpu[0], edge_index_cpu[1]):
            neighbors[src].add(dst)
        sample_contrastive_pairs.neighbors_dict = neighbors
    
    neighbors = sample_contrastive_pairs.neighbors_dict
    anchors, positives, negatives = [], [], []
    
    # Use vectorized operations where possible
    for i in range(num_nodes):
        if len(neighbors[i]) == 0:
            continue
        
        pos_candidates = list(neighbors[i])
        pos_sample = pos_candidates[torch.randint(len(pos_candidates), (1,)).item()]
        
        neg_candidates = list(set(range(num_nodes)) - neighbors[i] - {i})
        if len(neg_candidates) >= num_negatives:
            neg_sample = torch.tensor(neg_candidates)[torch.randperm(len(neg_candidates))[:num_negatives]].tolist()
            anchors.append(i)
            positives.append(pos_sample)
            negatives.append(neg_sample)
    
    return anchors, positives, negatives

def contrastive_loss(embeddings, anchors, positives, negatives, margin=0.5):
    """
    For each anchor node, the loss encourages the cosine similarity with a positive
    (neighbor) to be higher than with negatives (non-neighbors) by at least the margin.
    """
    loss_all = []
    for i, pos_idx, neg_idxs in zip(anchors, positives, negatives):
        anchor_emb = embeddings[i]
        pos_emb = embeddings[pos_idx]
        pos_sim = F.cosine_similarity(anchor_emb.unsqueeze(0), pos_emb.unsqueeze(0))
        
        neg_embs = embeddings[neg_idxs]
        neg_sim = F.cosine_similarity(anchor_emb.unsqueeze(0), neg_embs)
        
        # We want pos_sim - neg_sim > margin
        loss_per_negative = F.relu(margin - (pos_sim - neg_sim))
        loss_all.append(loss_per_negative.mean())
    if loss_all:
        return torch.stack(loss_all).mean()
    else:
        return torch.tensor(0.0, device=embeddings.device)

################################
# 8. TRAINING LOOP ON MPS/CPU   #
################################

optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
num_epochs = 1001

# Pre-compute neighbors dictionary before training loop
_ = sample_contrastive_pairs(data.edge_index, data.num_nodes, num_negatives=30)

model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    embeddings = model(data.x, data.edge_index, data.edge_attr)
    
    # Now uses cached neighbors dictionary
    anchors, positives, negatives = sample_contrastive_pairs(
        data.edge_index, data.num_nodes, num_negatives=30
    )
    
    loss = contrastive_loss(embeddings, anchors, positives, negatives, margin=0.5)
    loss.backward(retain_graph=True)
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")
