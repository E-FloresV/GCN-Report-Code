import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

#Demo de GCN
#Dhuaine Ramirez Leon, Erhuin Flores Vargas
#Proyecto de Investigacion 
#Curso Estructuras Discretas Para Informatica Universidad Nacional
#Docente: Carlos Loria
#Grupo: 03-8am

#Para instalar las librerias necesarias se deben instalar en el shell de windows con los siguientes comandos
#pip install torch
#pip install torch-geometric
#pip install matplotlib networkx

#Crear el grafo:Nodos, arcos, atributos y etiquetas.
# Definir los atributos de los nodos:[peso(g),categoria(comida=0, electrodomestico=1, ropa=2),precio($),cantidad]
x = torch.tensor([
    [25, 2, 7, 14],  # Camisa
    [30, 1, 4, 4],   # Plancha
    [22, 1, 9, 1],   # Televisor
    [28, 0, 3, 3],   # Cereal
    [65, 2, 23, 4],  # Pantalon
    [9, 0, 6, 10],   # atun
    [5400, 1, 120, 1]# microondas
], dtype=torch.float)

# Definir las conexiones entre nodos
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 4, 5, 6, 6],  # origen
    [4, 2, 6, 1, 6, 5, 0, 3, 1, 2]   # destino
], dtype=torch.long)

# Definir etiquetas (ocupaciones)
# 0 = comida, 1 = electrodomestio, 2 = ropa
y = torch.tensor([2, 1, 1, 0, 2, 0, 1], dtype=torch.long)

# Crear el grafo
data = Data(x=x, edge_index=edge_index, y=y)

# Crear máscaras de entrenamiento y prueba
# Esto se crea de manera generica por lo que se matiene una base y se juega a gusto con los parametros 
# y servira para la prediccion  
torch.manual_seed(42)
indices = torch.randperm(data.num_nodes)
train_size = int(data.num_nodes * 0.75)  # 75% entrenamiento, 25% prueba
train_idx = indices[:train_size]
test_idx = indices[train_size:]

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[train_idx] = True
data.test_mask[test_idx] = True

# Definir el modelo, heredado de torch.nn.Module.
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.num_node_features, 8)
        self.conv2 = GCNConv(8, 3)  # 3 categorías: comida, electrodomestico, ropa.
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Crear modelo, optimizador
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Entrenamiento
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

# Visualización
# Diccionario de nombres de los nodos:
node_names = {
    0: "Camisa",
    1: "Plancha",
    2: "Televisor",
    3: "Cereal",
    4: "Pantalón",
    5: "Atún",
    6: "Microondas"
}

# Diccionario de etiquetas reales para visualización:
labels = {0: "Comida", 1: "Electrodoméstico", 2: "Ropa"}

# Visualización para que el color represente la etiqueta, y cada producto(nodo) tenga su nombre:
# De esta manera se puede visualizar si se estan clasificando correctamente.
G = to_networkx(data, to_undirected=True)
plt.figure(figsize=(10, 7))

# Color por predicción(azul=comida, rojo= electrodomestico, verde=ropa)
color_map = ['red' if i == 1 else 'blue' if i == 0 else 'green' for i in pred.cpu()]

# Nombres correctos en el grafo utilizando el diccionario respectivo
nx.draw(
    G,
    node_color=color_map,
    labels=node_names,  # Aquí usamos los nombres reales de los productos
    with_labels=True,
    node_size=600,
    font_size=9
)

plt.title("Predicción de Categoría de Productos Varios por GCN")

# Opcional: Leyenda manual para saber qué color es qué
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color='blue', label='Comida'),
    mpatches.Patch(color='red', label='Electrodoméstico'),
    mpatches.Patch(color='green', label='Ropa')
]
plt.legend(handles=legend_handles)

plt.show()