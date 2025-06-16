import torch
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

# Cargar el dataset Karate Club
dataset = KarateClub()
data = dataset[0]

# Creamos manualmente las máscaras de entrenamiento y prueba
torch.manual_seed(42)  # Para reproducibilidad

# Mezclamos los índices de los nodos
indices = torch.randperm(data.num_nodes)

# Elegimos 80% para entrenamiento y 20% para prueba
train_size = int(data.num_nodes * 0.8)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

# Creamos las máscaras booleanas
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

# Asignamos True a los índices correspondientes
data.train_mask[train_idx] = True
data.test_mask[test_idx] = True

# Definir el modelo de la GCN
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Crear el modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Entrenar el modelo
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Evaluación
model.eval()
_, pred = model(data).max(dim=1)
correct = int((pred[data.test_mask] == data.y[data.test_mask]).sum())
acc = correct / int(data.test_mask.sum())
print(f'Precisión en test: {acc:.4f}')

# Visualización del grafo coloreado por predicción
import networkx as nx
from torch_geometric.utils import to_networkx

G = to_networkx(data, to_undirected=True)
plt.figure(figsize=(8, 6))
color_map = ['red' if i == 0 else 'blue' for i in pred.cpu()]
nx.draw(G, node_color=color_map, with_labels=True, node_size=500)
plt.title("Predicción de Comunidad por GCN")
plt.show()
