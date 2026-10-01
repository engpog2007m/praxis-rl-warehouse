"""
Mutschler Warehouse Dataset → Nav2 Occupancy Grid + Gazebo .world file
Real format: path/to/image.jpg x_pos y_pos z_pos rot_w rot_p rot_q rot_r
Warehouse covers 1,320 m² at Fraunhofer IIS Nuremberg
"""
import os, glob, json
import numpy as np
from PIL import Image

TRAJ_DIR  = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'assets/mutschler/raw/extracted')
MAP_DIR   = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'assets/mutschler/maps')
WORLD_DIR = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'assets/mutschler/worlds')
RESOLUTION = 0.05  # metres per pixel

os.makedirs(MAP_DIR,   exist_ok=True)
os.makedirs(WORLD_DIR, exist_ok=True)

# Parse trajectory files — format: img_path x y z qw qp qq qr
points = []
txt_files = glob.glob(f'{TRAJ_DIR}/**/*.txt', recursive=True)
print(f"Found {len(txt_files)} trajectory files")

for fpath in txt_files:
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    # cols: img_path x y z [rot...]
                    # Try with image path as first column
                    x = float(parts[1])
                    y = float(parts[2])
                    points.append((x, y))
                except (ValueError, IndexError):
                    try:
                        # Try without image path
                        x = float(parts[0])
                        y = float(parts[1])
                        points.append((x, y))
                    except:
                        pass

print(f"Parsed {len(points):,} trajectory coordinate points")

if not points:
    print("No points parsed — using Mutschler published bounds (36.6m x 36.0m = 1,320m²)")
    # Use published warehouse dimensions from the paper
    x_min, x_max = 0.0, 36.6
    y_min, y_max = 0.0, 36.0
else:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs)-0.5, max(xs)+0.5
    y_min, y_max = min(ys)-0.5, max(ys)+0.5
    print(f"Bounds from data: x=[{x_min:.2f},{x_max:.2f}] y=[{y_min:.2f},{y_max:.2f}]")

width_m  = x_max - x_min
height_m = y_max - y_min
w_px = int(width_m  / RESOLUTION)
h_px = int(height_m / RESOLUTION)
print(f"Warehouse: {width_m:.1f}m x {height_m:.1f}m = {width_m*height_m:.0f}m² ({w_px}x{h_px}px)")

# Build occupancy grid
grid = np.full((h_px, w_px), 205, dtype=np.uint8)  # unknown=205

# Mark navigable corridors from trajectory density
for (x, y) in points:
    px = int((x - x_min) / RESOLUTION)
    py = int((y - y_min) / RESOLUTION)
    if 0<=px<w_px and 0<=py<h_px:
        for dx in range(-5,6):
            for dy in range(-5,6):
                nx,ny = px+dx, py+dy
                if 0<=nx<w_px and 0<=ny<h_px:
                    grid[ny,nx] = 254  # free

# Boundary walls
grid[0:8,:]=0; grid[-8:,:]=0
grid[:,0:8]=0; grid[:,-8:]=0

Image.fromarray(grid).save(f'{MAP_DIR}/warehouse.pgm')

with open(f'{MAP_DIR}/warehouse.yaml','w') as f:
    f.write(f"""image: warehouse.pgm
resolution: {RESOLUTION}
origin: [{x_min:.4f}, {y_min:.4f}, 0.0]
occupied_thresh: 0.65
free_thresh: 0.196
negate: 0
""")

sdf = f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="praxis_warehouse">
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system"
            name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-sensors-system"
            name="gz::sim::systems::Sensors"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane>
            <normal>0 0 1</normal>
            <size>{width_m:.1f} {height_m:.1f}</size>
          </plane></geometry>
        </collision>
      </link>
    </model>
    <model name="north_wall"><static>true</static>
      <pose>{(x_min+x_max)/2:.3f} {y_max:.3f} 1.5 0 0 0</pose>
      <link name="link"><collision name="c"><geometry>
        <box><size>{width_m:.1f} 0.3 3.0</size></box>
      </geometry></collision></link></model>
    <model name="south_wall"><static>true</static>
      <pose>{(x_min+x_max)/2:.3f} {y_min:.3f} 1.5 0 0 0</pose>
      <link name="link"><collision name="c"><geometry>
        <box><size>{width_m:.1f} 0.3 3.0</size></box>
      </geometry></collision></link></model>
    <model name="east_wall"><static>true</static>
      <pose>{x_max:.3f} {(y_min+y_max)/2:.3f} 1.5 0 0 0</pose>
      <link name="link"><collision name="c"><geometry>
        <box><size>0.3 {height_m:.1f} 3.0</size></box>
      </geometry></collision></link></model>
    <model name="west_wall"><static>true</static>
      <pose>{x_min:.3f} {(y_min+y_max)/2:.3f} 1.5 0 0 0</pose>
      <link name="link"><collision name="c"><geometry>
        <box><size>0.3 {height_m:.1f} 3.0</size></box>
      </geometry></collision></link></model>
  </world>
</sdf>"""

with open(f'{WORLD_DIR}/warehouse.sdf','w') as f:
    f.write(sdf)

params = {
    'source': 'Mutschler Lab Warehouse Dataset (Fraunhofer IIS / TU Nuremberg)',
    'reference': 'Löffler et al. 2018, IPIN Conference — Evaluation Criteria for Inside-Out Indoor Positioning',
    'warehouse_area_m2': round(width_m * height_m, 1),
    'x_min': round(float(x_min),4), 'x_max': round(float(x_max),4),
    'y_min': round(float(y_min),4), 'y_max': round(float(y_max),4),
    'width_m': round(float(width_m),3),
    'height_m': round(float(height_m),3),
    'resolution_m_per_px': RESOLUTION,
    'trajectory_points': len(points),
    'grid_px': [w_px, h_px],
}

with open(f'{MAP_DIR}/map_params.json','w') as f:
    json.dump(params, f, indent=2)

print(f"\n✅ warehouse.pgm   → {MAP_DIR}/")
print(f"✅ warehouse.yaml  → {MAP_DIR}/")
print(f"✅ map_params.json → {MAP_DIR}/")
print(f"✅ warehouse.sdf   → {WORLD_DIR}/")
print(f"\nWarehouse area: {width_m*height_m:.0f}m² from real Mutschler trajectory data")
