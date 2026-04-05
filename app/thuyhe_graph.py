"""
Vietnam Inland Waterway Network via GeoJSON
Builds a NetworkX graph for routing and a cKDTree for spatial lookups.
"""

import json
import math
import heapq
import logging
from typing import Dict, List, Tuple
import networkx as nx
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# ─── Physics constants (must match model/main.py) ───────────────────────────
RHO  = 1000.0   # kg/m³
CD   = 0.8
SFC  = 200.0    # g/kWh
EF   = 3.2      # emission factor
DIESEL_DENSITY = 832.0  # g/L

ROUTE_COLORS = ["#58a6ff", "#3fb950", "#f78166", "#e3b341", "#a371f7"]


def haversine(lon1, lat1, lon2, lat2) -> float:
    """Calculate distance between two coords in km."""
    R = 6371.0
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2)**2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dLon / 2)**2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _physics_fuel(speed_kn: float, width_m: float, draft_m: float, distance_km: float) -> Tuple[float, float]:
    """Return (fuel_liters, co2_units) for one segment."""
    V     = max(speed_kn, 0.5) * 0.514444          # knots → m/s
    A     = max(width_m, 0.5) * max(draft_m, 0.3)  # frontal area m²
    R     = 0.5 * RHO * CD * A * V ** 2            # resistance N
    P_kw  = R * V / 1000.0                         # power kW
    rate  = SFC * P_kw                             # g/h
    t_h   = distance_km / (max(speed_kn, 0.5) * 1.852)
    fuel_g = rate * t_h
    fuel_L = fuel_g / DIESEL_DENSITY
    co2    = fuel_L * EF
    return round(fuel_L, 2), round(co2, 2)


class ThuyheGraph:
    def __init__(self, geojson_path: str):
        self.G = nx.Graph()
        self.node_list = []
        self.kdtree = None
        self._load_graph(geojson_path)

    def _load_graph(self, path: str):
        logger.info(f"Loading GeoJSON from {path}...")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            
        for feat in data.get("features", []):
            geom = feat.get("geometry")
            if not geom:
                continue
            
            # River name from properties 
            props = feat.get("properties", {})
            river = props.get("Ten") or "Kênh/Sông không tên"

            coords_list = geom["coordinates"]
            if geom["type"] == "LineString":
                coords_list = [coords_list]
            elif geom["type"] != "MultiLineString":
                continue
                
            for line in coords_list:
                for i in range(len(line) - 1):
                    # Round coords to avoid microscopic float mismatches
                    p1 = (round(line[i][0], 6), round(line[i][1], 6))
                    p2 = (round(line[i+1][0], 6), round(line[i+1][1], 6))
                    
                    dist = haversine(p1[0], p1[1], p2[0], p2[1])
                    if dist > 0:
                        self.G.add_edge(p1, p2, distance=dist, river=river)

        self.node_list = list(self.G.nodes())
        if self.node_list:
            self.kdtree = cKDTree(self.node_list)
        logger.info(f"Loaded ThuyheGraph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    def find_nearest_node(self, lon: float, lat: float) -> Tuple[float, float]:
        """Snap a given coordinate to the closest graph node."""
        if not self.kdtree:
            raise ValueError("Graph not initialized.")
        dist, idx = self.kdtree.query((lon, lat))
        return self.node_list[idx]

    def compute_path_stats(self, path: List[Tuple[float, float]], vessel: Dict) -> Dict:
        """Compute physics stats across a sequence of nodes."""
        speed = vessel.get("speed", 15.0)
        width = vessel.get("width", 5.0)
        draft = vessel.get("draft", 1.5)

        total_km = 0.0
        total_fuel = 0.0
        total_co2 = 0.0
        segments = []

        current_river = None
        current_segment_dist = 0.0
        start_coord = path[0]

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = self.G[u][v]
            dist = edge_data["distance"]
            river = edge_data["river"]
            
            if current_river is None:
                current_river = river

            if river != current_river:
                # Flush segment
                fuel, co2 = _physics_fuel(speed, width, draft, current_segment_dist)
                total_km += current_segment_dist
                total_fuel += fuel
                total_co2 += co2
                
                segments.append({
                    "from_name": f"{start_coord[1]:.4f}, {start_coord[0]:.4f}",
                    "to_name": f"{u[1]:.4f}, {u[0]:.4f}",
                    "distance_km": round(current_segment_dist, 2),
                    "river": current_river,
                    "fuel_L": fuel,
                    "co2": co2
                })
                
                current_river = river
                current_segment_dist = dist
                start_coord = u
            else:
                current_segment_dist += dist

        # Flush final segment
        fuel, co2 = _physics_fuel(speed, width, draft, current_segment_dist)
        total_km += current_segment_dist
        total_fuel += fuel
        total_co2 += co2
        
        segments.append({
            "from_name": f"{start_coord[1]:.4f}, {start_coord[0]:.4f}",
            "to_name": f"{path[-1][1]:.4f}, {path[-1][0]:.4f}",
            "distance_km": round(current_segment_dist, 2),
            "river": current_river,
            "fuel_L": fuel,
            "co2": co2
        })

        t_h = total_km / (max(speed, 0.5) * 1.852)
        return {
            "distance_km": round(total_km, 1),
            "time_hours":  round(t_h, 1),
            "fuel_L":      round(total_fuel, 1),
            "co2":         round(total_co2,  1),
            "segments":    segments,
        }

    def find_routes(self, src_lon: float, src_lat: float, dst_lon: float, dst_lat: float, vessel: Dict, k: int = 3) -> List[Dict]:
        """
        Find up to k shortest routes. Because simple_paths on a 110k graph can be very slow, 
        we'll just use single shortest path or limited K-shortest paths if nodes are connected.
        """
        n1 = self.find_nearest_node(src_lon, src_lat)
        n2 = self.find_nearest_node(dst_lon, dst_lat)

        if not nx.has_path(self.G, n1, n2):
            return []

        # Find shortest paths using networkx Yen's generator
        routes = []
        try:
            path_generator = nx.shortest_simple_paths(self.G, n1, n2, weight="distance")
            for rank, path in enumerate(path_generator):
                if rank >= k:
                    break
                stats = self.compute_path_stats(path, vessel)
                
                # Coords for Leaflet are [lat, lon]
                coords = [[p[1], p[0]] for p in path]
                
                label = ["Đề xuất tối ưu", "Tuyến phụ", "Tuyến nhánh"][rank] if rank < 3 else f"Tuyến {rank+1}"
                
                routes.append({
                    "id":         rank + 1,
                    "label":      label,
                    "color":      ROUTE_COLORS[rank % len(ROUTE_COLORS)],
                    "path":       path, # For reference
                    "path_names": [
                         f"Điểm phát ({n1[1]:.4f}, {n1[0]:.4f})", 
                         f"Điểm đến ({n2[1]:.4f}, {n2[0]:.4f})"
                    ],
                    "coordinates": coords,
                    "recommended": rank == 0,
                    **stats,
                })
        except (nx.NetworkXNoPath, nx.NetworkXError):
            pass

        return routes
