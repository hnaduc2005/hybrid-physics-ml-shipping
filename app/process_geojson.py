import json

with open("model/thuyhe.geojson", encoding="utf-8") as f:
    data = json.load(f)

features = data["features"]
print(f"Total features: {len(features)}")

points = set()
edges = 0
for feat in features:
    geom = feat["geometry"]
    if not geom: continue
    coords = geom["coordinates"]
    if geom["type"] == "MultiLineString":
        for line in coords:
            for p in line:
                points.add((p[0], p[1]))
            edges += len(line) - 1
    elif geom["type"] == "LineString":
        for p in coords:
            points.add((p[0], p[1]))
        edges += len(coords) - 1

print(f"Total unique points: {len(points)}")
print(f"Total edges: {edges}")
