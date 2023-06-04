import torch
from torch.nn import Linear, ReLU
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
import sys
import os
sys.path.append(os.path.abspath("."))
from load_data import MdDataset

class DualGCN(torch.nn.Module):
    def __init__(self, node_features, hidden_channels, output_channels):
        super(DualGCN, self).__init__()

        # GCN parts
        self.gcn = GCNConv(node_features, hidden_channels)

        # Output layer
        self.out = Linear(hidden_channels, output_channels)

        # Activation
        self.relu = ReLU()

    def forward(self, node_features, edge_index):
        # Obtain node-level embeddings 
        x = self.relu(self.protein_gcn(node_features, edge_index))

        # Use global_mean_pool to obtain graph-level embeddings
        protein_embed = global_mean_pool(x, protein_batch) 

        # Output layer
        out = self.out(protein_embed)

        return out


def train(model, loader, optimizer, device):
    model.train()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.protein, data.ligand)
        loss = torch.nn.functional.mse_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()

    return total_loss / len(loader.dataset)

def test(model, loader, device):
    model.eval()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data.protein, data.ligand)
        loss = torch.nn.functional.mse_loss(out, data.y)
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


#instantiate model
node_features = 11

hidden_channels = 32
output_channels = 1

model = DualGCN(node_features, 
    hidden_channels, 
    output_channels)

#train model
with open("/data/train_MD.txt", "r") as f:
    train_ids = [l.replace("\n", "") for l in f if l]
with open("/data/val_MD.txt", "r") as f:
    val_ids = [l for l in f]
with open("/data/test_MD.txt", "r") as f:
    test_ids = [l for l in f]
train_dataset = MdDataset("/data/MD.hdf5", train_ids, "r")

train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers = 32, pin_memory = True)
b = next(iter(train_dataloader))
print(b)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# for epoch in range(1, 201):
#     train_loss = train(model, train_loader, optimizer, device)
#     val_loss = test(model, val_loader, device)
#     print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# #test model
# test_loss = test(model, test_loader, device)
# print(f'Test Loss: {test_loss:.4f}')

# #save model
# torch.save(model.state_dict(), 'model.pt')