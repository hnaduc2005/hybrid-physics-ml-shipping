"""
FastAPI backend for Inland Waterway PIML Navigator using Thuyhe GeoJSON.
"""

import os, sys
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List

# resolve imports relative to project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.thuyhe_graph import ThuyheGraph

app = FastAPI(title="Vietnam Inland Waterway PIML Navigator", version="1.0.0")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
GEOJSON_PATH = os.path.join(ROOT, "model", "thuyhe.geojson")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Load graph at startup
print("Initializing ThuyheGraph. This might take a few moments...")
_graph = ThuyheGraph(GEOJSON_PATH)


# ─── Pydantic schemas ─────────────────────────────────────────────────────────
class VesselParams(BaseModel):
    speed:  float = Field(15.0, ge=1,  le=50,  description="Speed (knots)")
    length: float = Field(22.0, ge=5,  le=150, description="Length (m)")
    width:  float = Field(5.0,  ge=1,  le=30,  description="Width / beam (m)")
    draft:  float = Field(1.5,  ge=0.3, le=8,  description="Draft (m)")
    vessel_type: str = Field("PASSAGEIRO/CARGA GERAL")


class RouteRequest(BaseModel):
    origin:      List[float]    # [lon, lat]
    destination: List[float]    # [lon, lat]
    vessel:      VesselParams = VesselParams()
    num_routes:  int = Field(3, ge=1, le=5)


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/network")
async def get_network():
    """Return waterway network as GeoJSON directly from file for Leaflet rendering."""
    return FileResponse(GEOJSON_PATH, media_type="application/json")


@app.post("/api/find-routes")
async def find_routes(req: RouteRequest):
    """Find candidate routes between origin and destination coordinates."""
    if len(req.origin) != 2 or len(req.destination) != 2:
        raise HTTPException(400, "Origin and destination must be [lon, lat].")
        
    src_lon, src_lat = req.origin
    dst_lon, dst_lat = req.destination

    vessel = req.vessel.model_dump()
    routes = _graph.find_routes(src_lon, src_lat, dst_lon, dst_lat, vessel, req.num_routes)

    if not routes:
        raise HTTPException(404, "No path found between selected locations. They might be in different disconnected components of the river network.")

    # Snap to actual nodes for response
    n1 = _graph.find_nearest_node(src_lon, src_lat)
    n2 = _graph.find_nearest_node(dst_lon, dst_lat)

    return {
        "origin":      {"lon": n1[0], "lat": n1[1]},
        "destination": {"lon": n2[0], "lat": n2[1]},
        "vessel":      vessel,
        "routes":      routes,
    }
