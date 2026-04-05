/* ═══════════════════════════════════════════════════════════════════════════
   PIML Navigator – Frontend Logic (Vietnam GeoJSON)
   Leaflet map + API integration + UI interactions
   ═══════════════════════════════════════════════════════════════════════════ */

'use strict';

// ─── App state ───────────────────────────────────────────────────────────────
const state = {
  origin:       null,        // [lon, lat]
  destination:  null,        // [lon, lat]
  routes:       [],
  activeRoute:  0,
  clickMode:    null,        // 'origin' | 'dest' | null
  layers: {
    network:    null,
    routes:     [],
    endpointMarkers: {
      origin: null,
      dest: null
    },
  }
};

// ─── Map initialisation ──────────────────────────────────────────────────────
const map = L.map('map', {
  center: [9.5, 105.5],      // Centered on Mekong Delta / Southern Vietnam
  zoom: 8,
  zoomControl: false,
  attributionControl: false,
  preferCanvas: true         // Highly recommended for large GeoJSON (110k segments)
});
L.control.zoom({ position: 'bottomright' }).addTo(map);

L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
  maxZoom: 19,
  subdomains: 'abcd',
}).addTo(map);

L.control.attribution({ prefix: '© CartoDB | PIML Navigator' }).addTo(map);

// ─── Custom icons ────────────────────────────────────────────────────────────
function makeEndpointIcon(emoji) {
  return L.divIcon({
    className: '',
    html: `<div style="font-size:1.5rem;line-height:1;filter:drop-shadow(0 2px 4px #000a);">${emoji}</div>`,
    iconAnchor: [12, 24],
  });
}

// ─── Utility ─────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

function setHidden(el, hidden) {
  if (typeof el === 'string') el = $(el);
  el.classList.toggle('hidden', hidden);
}

function fmtNum(n, dec = 1) {
  return Number(n).toLocaleString('vi-VN', { maximumFractionDigits: dec });
}

function fmtCoord(coord) {
  if (!coord) return "Chưa chọn (Nhấp vào bản đồ)";
  return `${coord[1].toFixed(5)}, ${coord[0].toFixed(5)}`;
}

// ─── Physics preview ─────────────────────────────────────────────────────────
function updatePhysicsPreview() {
  const speed = parseFloat($('p-speed').value)  || 15;
  const width = parseFloat($('p-width').value)  || 5;
  const draft = parseFloat($('p-draft').value)  || 1.5;

  const V    = speed * 0.514444;
  const A    = width * draft;
  const R    = 0.5 * 1000 * 0.8 * A * V * V;
  const P_kw = R * V / 1000;
  const rate = 200 * P_kw / 832;  // L/h

  $('preview-power').textContent = P_kw.toFixed(1) + ' kW';
  $('preview-rate').textContent  = rate.toFixed(2) + ' L/h';
}

['p-speed', 'p-width', 'p-draft'].forEach(id =>
  $(id).addEventListener('input', updatePhysicsPreview)
);
updatePhysicsPreview();

// ─── Load network ────────────────────────────────────────────────────────────
async function loadNetwork() {
  try {
    const geojson = await fetch('/api/network').then(r => r.json());

    // Draw waterway lines (large dataset)
    state.layers.network = L.geoJSON(geojson, {
      style: () => ({
        color: '#1d4a8a',
        weight: 1.5,
        opacity: 0.6,
        dashArray: null,
      }),
      onEachFeature(feature, layer) {
        const p = feature.properties;
        const name = p.Ten || "Chưa có tên";
        layer.bindTooltip(`<strong>${name}</strong><br/>${fmtNum(p.Chieu_dai, 0)} m`, {
          sticky: true, className: 'ww-tooltip',
          direction: 'top',
        });
      }
    }).addTo(map);

    // Fit bounds to entire network
    map.fitBounds(state.layers.network.getBounds(), { padding: [20, 20] });

  } catch (err) {
    console.error("Lỗi khi tải mạng lưới geojson:", err);
  }
}

// ─── Selection logic ─────────────────────────────────────────────────────────
function setClickMode(mode) {
  state.clickMode = mode;
  $('coord-origin').classList.toggle('active', mode === 'origin');
  $('coord-dest').classList.toggle('active', mode === 'dest');

  if (mode === 'origin') {
    $('hint-text').innerHTML = 'Nhấp vào bản đồ để chọn <strong>điểm xuất phát</strong>';
    setHidden('map-hint', false);
  } else if (mode === 'dest') {
    $('hint-text').innerHTML = 'Nhấp vào bản đồ để chọn <strong>điểm đến</strong>';
    setHidden('map-hint', false);
  } else {
    setHidden('map-hint', true);
  }
}

// UI bindings for click mode
$('coord-origin').addEventListener('click', () => setClickMode('origin'));
$('coord-dest').addEventListener('click', () => setClickMode('dest'));
$('map-hint').addEventListener('click', () => setClickMode(null));

// Map click
map.on('click', e => {
  if (!state.clickMode) return;
  const coord = [e.latlng.lng, e.latlng.lat];

  if (state.clickMode === 'origin') {
    state.origin = coord;
    $('coord-origin').textContent = fmtCoord(coord);
    setClickMode('dest');
  } else {
    state.destination = coord;
    $('coord-dest').textContent = fmtCoord(coord);
    setClickMode(null);
  }
  
  refreshEndpointMarkers();
  checkCanFind();
});

function refreshEndpointMarkers() {
  if (state.layers.endpointMarkers.origin) {
    map.removeLayer(state.layers.endpointMarkers.origin);
  }
  if (state.layers.endpointMarkers.dest) {
    map.removeLayer(state.layers.endpointMarkers.dest);
  }

  if (state.origin) {
    state.layers.endpointMarkers.origin = L.marker([state.origin[1], state.origin[0]], { 
      icon: makeEndpointIcon('🟢') 
    }).addTo(map);
  }
  if (state.destination) {
    state.layers.endpointMarkers.dest = L.marker([state.destination[1], state.destination[0]], { 
      icon: makeEndpointIcon('🔴') 
    }).addTo(map);
  }
}

function checkCanFind() {
  $('btn-find').disabled = !(state.origin && state.destination);
}

// Swap
$('btn-swap').addEventListener('click', () => {
  const temp = state.origin;
  state.origin = state.destination;
  state.destination = temp;
  
  $('coord-origin').textContent = fmtCoord(state.origin);
  $('coord-dest').textContent = fmtCoord(state.destination);
  
  refreshEndpointMarkers();
  checkCanFind();
});

// ─── Find routes ─────────────────────────────────────────────────────────────
$('btn-find').addEventListener('click', findRoutes);

async function findRoutes() {
  if (!state.origin || !state.destination) return;

  // Loading state
  $('btn-label').textContent = 'Đang tìm...';
  setHidden('btn-spinner', false);
  $('btn-find').disabled = true;

  const payload = {
    origin:      state.origin,
    destination: state.destination,
    num_routes:  3,
    vessel: {
      speed:       parseFloat($('p-speed').value),
      length:      parseFloat($('p-length').value),
      width:       parseFloat($('p-width').value),
      draft:       parseFloat($('p-draft').value),
      vessel_type: 'PASSAGEIRO/CARGA GERAL',
    }
  };

  try {
    const resp = await fetch('/api/find-routes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const err = await resp.json();
      alert('Lỗi: ' + (err.detail || 'Không tìm được tuyến đường.'));
      return;
    }
    const data = await resp.json();
    
    // The backend returns the "snapped" coordinates. Update the UI.
    state.origin = [data.origin.lon, data.origin.lat];
    state.destination = [data.destination.lon, data.destination.lat];
    $('coord-origin').textContent = fmtCoord(state.origin);
    $('coord-dest').textContent = fmtCoord(state.destination);
    refreshEndpointMarkers();

    state.routes = data.routes;
    renderRoutes(data.routes);
    state.activeRoute = 0;
    activateRoute(0);
    updateLegend(data.routes);

  } catch (e) {
    alert('Không kết nối được với máy chủ. Vui lòng thử lại sau.');
    console.error(e);
  } finally {
    $('btn-label').textContent = 'Tìm tuyến đường';
    setHidden('btn-spinner', true);
    $('btn-find').disabled = false;
  }
}

// ─── Render route cards ──────────────────────────────────────────────────────
function renderRoutes(routes) {
  clearRouteLayers();
  setHidden('results-panel', false);
  const container = $('route-cards');
  container.innerHTML = '';

  routes.forEach((route, idx) => {
    // Draw polyline on map
    const line = L.polyline(route.coordinates, {
      color:   route.color,
      weight:  idx === 0 ? 5 : 3,
      opacity: idx === 0 ? 0.9 : 0.45,
      dashArray: idx === 0 ? null : '6 4',
    }).addTo(map);
    line.on('click', () => activateRoute(idx));
    state.layers.routes.push(line);

    // Route path summary
    const pathSummary = route.path_names.join(' → ');

    // Card HTML
    const card = document.createElement('div');
    card.className = 'route-card' + (idx === 0 ? ' active' : '');
    card.style.setProperty('--color', route.color);
    card.dataset.idx = idx;
    card.innerHTML = `
      <div class="card-header">
        <span class="card-label">${route.label}</span>
        <span class="card-path" title="${pathSummary}">${pathSummary}</span>
      </div>
      <div class="card-stats">
        <div class="stat-item">
          <span class="stat-label">Khoảng cách</span>
          <span class="stat-val">${fmtNum(route.distance_km, 0)}<span class="stat-unit"> km</span></span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Thời gian</span>
          <span class="stat-val">${fmtNum(route.time_hours, 1)}<span class="stat-unit"> h</span></span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Nhiên liệu</span>
          <span class="stat-val">${fmtNum(route.fuel_L, 0)}<span class="stat-unit"> L</span></span>
        </div>
        <div class="stat-item">
          <span class="stat-label">CO₂</span>
          <span class="stat-val">${fmtNum(route.co2, 0)}<span class="stat-unit"> đv</span></span>
        </div>
      </div>
      <button class="card-detail-btn" data-idx="${idx}">Xem chi tiết đoạn đường →</button>
    `;
    card.addEventListener('click', e => {
      if (e.target.classList.contains('card-detail-btn')) {
        showRouteDetail(routes[idx]);
      } else {
        activateRoute(idx);
      }
    });
    container.appendChild(card);
  });

  // Fit map to first route
  if (routes[0]?.coordinates?.length) {
    map.fitBounds(L.latLngBounds(routes[0].coordinates), { padding: [40, 40] });
  }
}

function activateRoute(idx) {
  state.activeRoute = idx;
  // Update map layers
  state.layers.routes.forEach((line, i) => {
    line.setStyle({
      weight:    i === idx ? 5 : 3,
      opacity:   i === idx ? 0.9 : 0.35,
      dashArray: i === idx ? null : '6 4',
    });
    if (i === idx) line.bringToFront();
  });
  // Update cards
  document.querySelectorAll('.route-card').forEach((card, i) =>
    card.classList.toggle('active', i === idx)
  );
}

// ─── Route detail modal ───────────────────────────────────────────────────────
function showRouteDetail(route) {
  const rows = route.segments.map(s => `
    <tr>
      <td>${s.from_name}</td>
      <td>${s.to_name}</td>
      <td>${s.river}</td>
      <td style="text-align:right">${fmtNum(s.distance_km, 2)}</td>
      <td style="text-align:right">${fmtNum(s.fuel_L, 1)}</td>
      <td style="text-align:right">${fmtNum(s.co2, 1)}</td>
    </tr>
  `).join('');

  $('modal-content').innerHTML = `
    <div class="modal-title" style="color:${route.color}">
      ${route.label} — Tuyến đường nội địa (<span style="font-size:0.8em;color:#888">${route.segments.length} chặng</span>)
    </div>
    <div class="summary-bar">
      <div class="summary-item">
        <div class="si-val">${fmtNum(route.distance_km, 1)}</div>
        <div class="si-label">km</div>
      </div>
      <div class="summary-item">
        <div class="si-val">${fmtNum(route.time_hours, 1)}</div>
        <div class="si-label">giờ</div>
      </div>
      <div class="summary-item">
        <div class="si-val">${fmtNum(route.fuel_L, 0)}</div>
        <div class="si-label">Lít nhiên liệu</div>
      </div>
      <div class="summary-item">
        <div class="si-val">${fmtNum(route.co2, 0)}</div>
        <div class="si-label">CO₂ (đv)</div>
      </div>
    </div>
    <table class="seg-table">
      <thead>
        <tr>
          <th>Từ (Tọa độ)</th><th>Đến (Tọa độ)</th><th>Kênh / Lưu Vực</th>
          <th style="text-align:right">km</th>
          <th style="text-align:right">Nhiên liệu (L)</th>
          <th style="text-align:right">CO₂</th>
        </tr>
      </thead>
      <tbody>
        ${rows}
        <tr class="total-row">
          <td colspan="3"><strong>Tổng cộng</strong></td>
          <td style="text-align:right"><strong>${fmtNum(route.distance_km, 1)}</strong></td>
          <td style="text-align:right"><strong>${fmtNum(route.fuel_L, 1)}</strong></td>
          <td style="text-align:right"><strong>${fmtNum(route.co2, 1)}</strong></td>
        </tr>
      </tbody>
    </table>
  `;
  setHidden('modal-overlay', false);
}
$('modal-close').addEventListener('click', () => setHidden('modal-overlay', true));
$('modal-overlay').addEventListener('click', e => {
  if (e.target === $('modal-overlay')) setHidden('modal-overlay', true);
});

// ─── Legend ───────────────────────────────────────────────────────────────────
function updateLegend(routes) {
  const legend = $('map-legend');
  legend.style.display = 'block';
  legend.innerHTML = routes.map(r => `
    <div class="legend-item">
      <div class="legend-line" style="background:${r.color}"></div>
      <span style="font-size:.74rem;color:#e6edf3">${r.label}</span>
      <span style="font-size:.72rem;color:#8b949e;margin-left:auto;padding-left:12px">${fmtNum(r.fuel_L,0)} L</span>
    </div>
  `).join('');
}

// ─── Cleanup helpers ──────────────────────────────────────────────────────────
function clearRouteLayers() {
  state.layers.routes.forEach(l => map.removeLayer(l));
  state.layers.routes = [];
  $('map-legend').style.display = 'none';
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
(async () => {
  await loadNetwork();
  setClickMode('origin');
})();
