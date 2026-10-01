import json, os

BASE = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'assets')

class DataRegistry:
    def __init__(self):
        mp = f'{BASE}/mutschler/maps/map_params.json'
        if os.path.exists(mp):
            with open(mp) as f: m = json.load(f)
            self.arena_width  = m['width_m']
            self.arena_height = m['height_m']
            self.x_min = m['x_min']; self.x_max = m['x_max']
            self.y_min = m['y_min']; self.y_max = m['y_max']
            print(f"[DataRegistry] Mutschler: {self.arena_width}m x {self.arena_height}m "
                  f"({m['warehouse_area_m2']}m²)")
        else:
            self.arena_width = self.arena_height = 15.0
            self.x_min = self.y_min = 0.0
            self.x_max = self.y_max = 15.0
            print("[DataRegistry] Mutschler not found — default")

        sp = f'{BASE}/sku110k/density_params.json'
        if os.path.exists(sp):
            with open(sp) as f: sku = json.load(f)
            self.noise_sigmas   = sku['training_sigmas_used']
            self.sku110k_images = sku.get('total_images', 0)
            stats = sku['density_statistics']
            print(f"[DataRegistry] SKU110K: {self.sku110k_images:,} images | "
                  f"mean={stats['mean']} objects/image | sigmas={self.noise_sigmas}")
        else:
            self.noise_sigmas   = [0.00, 0.05, 0.15, 0.30]
            self.sku110k_images = 0
            print("[DataRegistry] SKU110K not found — default sigmas")

        yp = f'{BASE}/ycb/ycb_task_registry.json'
        if os.path.exists(yp):
            with open(yp) as f: ycb = json.load(f)
            raw = ycb['task_registry']
            self.tasks = {int(k): v for k, v in raw.items()}
            print(f"[DataRegistry] YCB: {len(self.tasks)} task profiles | "
                  f"source={ycb['source']}")
        else:
            self.tasks = {
                1:    {'name':'standard_box',     'is_heavy':False,'is_fragile':False,'weight_kg':5.0},
                2:    {'name':'lightweight_pkg',  'is_heavy':False,'is_fragile':False,'weight_kg':2.0},
                3:    {'name':'heavy_pallet',     'is_heavy':True, 'is_fragile':False,'weight_kg':45.0},
                4:    {'name':'fragile_glassware','is_heavy':False,'is_fragile':True, 'weight_kg':3.5},
                1001: {'name':'heavy_fragile',    'is_heavy':True, 'is_fragile':True, 'weight_kg':30.0},
            }
            print("[DataRegistry] YCB not found — hardcoded fallback")

    def get_task(self, task_id):
        return self.tasks.get(task_id, self.tasks.get(1))

    def get_mass(self, task_id):
        t = self.get_task(task_id)
        return t.get('weight_kg', t.get('weight', 5.0))

    def get_noise_sigma(self, level_index):
        return self.noise_sigmas[level_index % len(self.noise_sigmas)]
