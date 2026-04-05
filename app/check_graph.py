import json
import networkx as nx
from scipy.spatial import cKDTree
import math

def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

with open("model/thuyhe.geojson", encoding="utf-8") as f:
    data = json.load(f)

G = nx.Graph()

for feat in data["features"]:
    geom = feat.get("geometry")
    if not geom: continue
    coords_list = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
    
    for line in coords_list:
        for i in range(len(line) - 1):
            p1 = (round(line[i][0], 6), round(line[i][1], 6))
            p2 = (round(line[i+1][0], 6), round(line[i+1][1], 6))
            
            dist = haversine(p1[0], p1[1], p2[0], p2[1])
            G.add_edge(p1, p2, distance=dist)

components = list(nx.connected_components(G))
sizes = sorted([len(c) for c in components], reverse=True)

with open("app/check_graph.txt", "w", encoding="utf-8") as f:
    f.write(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}\n")
    f.write(f"Connected components: {len(components)}\n")
    f.write(f"Top component sizes: {sizes[:10]}\n")
