"""
Amazon Inland Waterway Network
Ports, edges, and multi-route path-finding algorithms.
"""

import heapq
import math
from typing import Dict, List, Tuple, Any

# ─── Physics constants (must match model/main.py) ───────────────────────────
RHO  = 1000.0   # kg/m³
CD   = 0.8
SFC  = 200.0    # g/kWh
EF   = 3.2      # emission factor
DIESEL_DENSITY = 832.0  # g/L

# ─── Port definitions ────────────────────────────────────────────────────────
PORTS: Dict[str, Dict] = {
    "BELEM":       {"name": "Belém",        "lat": -1.4558,  "lon": -48.5044, "state": "PA"},
    "BREVES":      {"name": "Breves",       "lat": -1.6833,  "lon": -50.4806, "state": "PA"},
    "CAMETA":      {"name": "Cametá",       "lat": -2.2447,  "lon": -49.4961, "state": "PA"},
    "ABAETETUBA":  {"name": "Abaetetuba",   "lat": -1.7228,  "lon": -48.8789, "state": "PA"},
    "TUCURUI":     {"name": "Tucuruí",      "lat": -3.7664,  "lon": -49.6736, "state": "PA"},
    "MACAPA":      {"name": "Macapá",       "lat":  0.0349,  "lon": -51.0694, "state": "AP"},
    "SANTANA":     {"name": "Santana",      "lat": -0.0583,  "lon": -51.1786, "state": "AP"},
    "SANTAREM":    {"name": "Santarém",     "lat": -2.4345,  "lon": -54.7082, "state": "PA"},
    "ITAITUBA":    {"name": "Itaituba",     "lat": -4.2756,  "lon": -55.9833, "state": "PA"},
    "OBIDOS":      {"name": "Óbidos",       "lat": -1.9152,  "lon": -55.5169, "state": "PA"},
    "ORIXIMINA":   {"name": "Oriximiná",    "lat": -1.7653,  "lon": -55.8617, "state": "PA"},
    "JURUTI":      {"name": "Juruti",       "lat": -2.1528,  "lon": -56.0875, "state": "PA"},
    "PARINTINS":   {"name": "Parintins",    "lat": -2.6275,  "lon": -56.7361, "state": "AM"},
    "MAUES":       {"name": "Maués",        "lat": -3.3869,  "lon": -57.7194, "state": "AM"},
    "MANAUS":      {"name": "Manaus",       "lat": -3.1190,  "lon": -60.0217, "state": "AM"},
    "BORBA":       {"name": "Borba",        "lat": -4.3869,  "lon": -59.5944, "state": "AM"},
    "MANICORE":    {"name": "Manicoré",     "lat": -5.8100,  "lon": -61.2800, "state": "AM"},
    "HUMAITA":     {"name": "Humaitá",      "lat": -7.5058,  "lon": -63.0219, "state": "AM"},
    "PORTO_VELHO": {"name": "Porto Velho",  "lat": -8.7612,  "lon": -63.9004, "state": "RO"},
    "COARI":       {"name": "Coari",        "lat": -4.0853,  "lon": -63.1408, "state": "AM"},
    "TEFE":        {"name": "Tefé",         "lat": -3.3653,  "lon": -64.7161, "state": "AM"},
    "TABATINGA":   {"name": "Tabatinga",    "lat": -4.2544,  "lon": -69.9378, "state": "AM"},
}

# ─── Edge definitions: (A, B, km, river_name) ───────────────────────────────
EDGES_RAW: List[Tuple] = [
    # Para / Marajó channels
    ("BELEM",      "BREVES",      185,  "Canal do Marajó"),
    ("BELEM",      "MACAPA",      420,  "Foz do Amazonas"),
    ("BELEM",      "ABAETETUBA",   55,  "Rio Pará"),
    ("BELEM",      "CAMETA",      210,  "Rio Tocantins"),
    ("BREVES",     "SANTANA",     195,  "Furo do Breves"),
    ("BREVES",     "CAMETA",      120,  "Baía de Marajó"),
    ("ABAETETUBA", "CAMETA",       90,  "Rio Tocantins"),
    ("CAMETA",     "TUCURUI",     150,  "Rio Tocantins"),
    ("MACAPA",     "SANTANA",      15,  "Rio Amazonas"),
    # Middle Amazon
    ("SANTANA",    "OBIDOS",      580,  "Rio Amazonas"),
    ("OBIDOS",     "JURUTI",      100,  "Rio Amazonas"),
    ("OBIDOS",     "ORIXIMINA",    60,  "Rio Trombetas"),
    ("JURUTI",     "SANTAREM",    120,  "Rio Amazonas"),
    ("ORIXIMINA",  "SANTAREM",    250,  "Rio Amazonas"),
    ("SANTAREM",   "ITAITUBA",    560,  "Rio Tapajós"),
    ("SANTAREM",   "PARINTINS",   380,  "Rio Amazonas"),
    # Upper Amazon
    ("PARINTINS",  "MAUES",       250,  "Rio Andirá"),
    ("PARINTINS",  "MANAUS",      450,  "Rio Amazonas"),
    ("MAUES",      "MANAUS",      350,  "Rio Maués-Mirim"),
    ("MANAUS",     "COARI",       400,  "Rio Solimões"),
    ("COARI",      "TEFE",        200,  "Rio Solimões"),
    ("TEFE",       "TABATINGA",  1050,  "Rio Solimões"),
    # Madeira tributary
    ("MANAUS",     "BORBA",       160,  "Rio Madeira"),
    ("BORBA",      "MANICORE",    130,  "Rio Madeira"),
    ("MANICORE",   "HUMAITA",     200,  "Rio Madeira"),
    ("HUMAITA",    "PORTO_VELHO", 280,  "Rio Madeira"),
    # Cross connections
    ("MAUES",      "BORBA",       180,  "Rio Maués-Açu"),
    ("PARINTINS",  "BORBA",       350,  "Rio Amazonas-Madeira"),
    ("SANTAREM",   "OBIDOS",      130,  "Rio Amazonas"),
    ("BREVES",     "TUCURUI",     200,  "Canal interno"),
]

# Route display colours (one per candidate rank)
ROUTE_COLORS = ["#58a6ff", "#3fb950", "#f78166", "#e3b341", "#a371f7"]


def _physics_fuel(speed_kn: float, width_m: float, draft_m: float,
                  distance_km: float) -> Tuple[float, float]:
    """Return (fuel_liters, co2_units) for one segment."""
    V     = max(speed_kn, 0.5) * 0.514444          # knots → m/s
    A     = max(width_m, 0.5) * max(draft_m, 0.3)  # frontal area m²
    R     = 0.5 * RHO * CD * A * V ** 2            # resistance N
    P_kw  = R * V / 1000.0                          # power kW
    rate  = SFC * P_kw                              # g/h
    t_h   = distance_km / (max(speed_kn, 0.5) * 1.852)
    fuel_g = rate * t_h
    fuel_L = fuel_g / DIESEL_DENSITY
    co2    = fuel_L * EF
    return round(fuel_L, 2), round(co2, 2)


class WaterwayGraph:
    def __init__(self):
        # adj[u][v] = {"distance": km, "river": str}
        self.adj: Dict[str, Dict[str, Dict]] = {p: {} for p in PORTS}
        for a, b, km, river in EDGES_RAW:
            self.adj[a][b] = {"distance": km, "river": river}
            self.adj[b][a] = {"distance": km, "river": river}

    # ── k-shortest simple paths (priority-queue approach) ───────────────────
    def k_shortest_paths(self, src: str, dst: str, k: int = 5) -> List[Tuple]:
        """
        Return up to k shortest simple paths as [(total_dist, [node,...]), ...].
        Uses a min-heap; guarantees no repeated nodes per path.
        """
        if src not in self.adj or dst not in self.adj:
            return []
        heap   = [(0.0, [src])]
        found  = []
        visits = {}          # node → how many times popped
        while heap and len(found) < k:
            cost, path = heapq.heappop(heap)
            node = path[-1]
            visits[node] = visits.get(node, 0) + 1
            if visits[node] > k:
                continue
            if node == dst:
                found.append((cost, path))
                continue
            for nb, edata in self.adj.get(node, {}).items():
                if nb not in path:
                    heapq.heappush(heap, (cost + edata["distance"], path + [nb]))
        return found

    # ── Route statistics ────────────────────────────────────────────────────
    def compute_path_stats(self, path: List[str], vessel: Dict) -> Dict:
        """Compute total distance, fuel, CO2, time for a list of port codes."""
        speed   = vessel.get("speed", 15.0)
        width   = vessel.get("width", 5.0)
        draft   = vessel.get("draft", 1.5)

        total_km   = 0.0
        total_fuel = 0.0
        total_co2  = 0.0
        segments   = []

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            km   = self.adj[u][v]["distance"]
            river = self.adj[u][v]["river"]
            fuel, co2 = _physics_fuel(speed, width, draft, km)
            total_km   += km
            total_fuel += fuel
            total_co2  += co2
            segments.append({
                "from": u, "from_name": PORTS[u]["name"],
                "to":   v, "to_name":   PORTS[v]["name"],
                "distance_km": km,
                "river":  river,
                "fuel_L": fuel,
                "co2":    co2,
            })

        t_h = total_km / (max(speed, 0.5) * 1.852)
        return {
            "distance_km": round(total_km, 1),
            "time_hours":  round(t_h, 1),
            "fuel_L":      round(total_fuel, 1),
            "co2":         round(total_co2,  1),
            "segments":    segments,
        }

    # ── Main API ─────────────────────────────────────────────────────────────
    def find_routes(self, src: str, dst: str, vessel: Dict, k: int = 4) -> List[Dict]:
        """Return up to k route dicts with statistics and display metadata."""
        candidates = self.k_shortest_paths(src, dst, k + 1)
        if not candidates:
            return []

        labels = ["Đề xuất", "Tiết kiệm", "Nhanh nhất", "Tuyến phụ"]
        routes = []
        for rank, (dist, path) in enumerate(candidates[:k]):
            stats = self.compute_path_stats(path, vessel)
            coords = [[PORTS[p]["lat"], PORTS[p]["lon"]] for p in path]
            routes.append({
                "id":         rank + 1,
                "label":      labels[rank] if rank < len(labels) else f"Tuyến {rank+1}",
                "color":      ROUTE_COLORS[rank % len(ROUTE_COLORS)],
                "path":       path,
                "path_names": [PORTS[p]["name"] for p in path],
                "coordinates": coords,
                **stats,
            })

        # Recommend the one with minimum fuel (often same as shortest, but
        # label makes UX clearer)
        best_idx = min(range(len(routes)), key=lambda i: routes[i]["fuel_L"])
        routes[best_idx]["label"] = "Tối ưu ✦"
        routes[best_idx]["recommended"] = True
        for r in routes:
            if "recommended" not in r:
                r["recommended"] = False
        return routes

    # ── GeoJSON for Leaflet ──────────────────────────────────────────────────
    def get_network_geojson(self) -> Dict:
        features = []
        seen = set()
        for a, b, km, river in EDGES_RAW:
            key = tuple(sorted([a, b]))
            if key in seen:
                continue
            seen.add(key)
            coords = [
                [PORTS[a]["lon"], PORTS[a]["lat"]],
                [PORTS[b]["lon"], PORTS[b]["lat"]],
            ]
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"river": river, "km": km, "from": a, "to": b},
            })
        # Port markers
        for code, info in PORTS.items():
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point",
                             "coordinates": [info["lon"], info["lat"]]},
                "properties": {"code": code, "name": info["name"],
                               "state": info["state"]},
            })
        return {"type": "FeatureCollection", "features": features}
