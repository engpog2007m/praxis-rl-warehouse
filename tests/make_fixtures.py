#!/usr/bin/env python3
"""Build self-test fixtures in a target directory.

  mock_praxis/          a miniature Praxis tree containing every edge case
                        the collector must handle
  corrected_world.sdf   a two-robot world in the corrected physics state
  synth/data/*.csv      result files whose binary-outcome counts equal those
                        implied by the reported rates. Mann-Whitney p-values
                        for binary outcomes depend only on counts, so these
                        must reproduce the reported p-values exactly.
"""
import os
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
from world_physics import BASE_FIXED, MASS_FIXED, WHEEL_FIXED  # noqa: E402


def mock_tree(root):
    m = root / 'mock_praxis'
    for d in ['src/simulation', 'scripts', 'worlds', 'assets/sku110k',
              'models', 'data', 'results']:
        (m / d).mkdir(parents=True, exist_ok=True)
    (m / 'src/simulation/gazebo_env.py').write_text(
        "import os, sys\nimport numpy as np\n"
        "sys.path.insert(0, os.path.expanduser('~/Praxis/src'))\n"
        "class GazeboWarehouseEnv:\n"
        "    GOAL_OFFSETS = {\n"
        "        1:    np.array([3.0, 0.0]),\n        2:    np.array([3.5, 0.0]),\n"
        "        3:    np.array([2.5, 0.0]),\n        4:    np.array([1.2, 0.0]),\n"
        "        1001: np.array([1.5, 0.0]),\n    }\n"
        "    def __init__(self, max_steps=200):\n        pass\n"
        "    def f(self, task, speed):\n"
        "        return (task['is_fragile'] and speed > 0.5, task['is_heavy'] and speed > 0.7)\n")
    (m / 'src/simulation/gazebo_env_lidar.py').write_text(
        "import numpy as np\nclass GazeboWarehouseEnv:\n"
        "    ARRIVAL_RADIUS    = 0.5\n    FRAGILE_THRESHOLD = 0.5\n    HEAVY_THRESHOLD   = 0.7\n"
        "    GOAL_OFFSETS = {\n"
        "        1:    np.array([2.0, 0.0]),\n        2:    np.array([2.0, 0.0]),\n"
        "        3:    np.array([2.0, 0.0]),\n        4:    np.array([2.0, 0.0]),\n"
        "        1001: np.array([2.0, 0.0]),\n    }\n"
        "    def __init__(self, max_steps=500, collision_dist=0.30):\n        pass\n"
        "    def reset(self):\n        off = self.np_random.uniform(-0.75, 0.75)\n")
    (m / 'src/simulation/data_registry.py').write_text('x = 1\n')
    (m / 'src/simulation/warehouse_env.py').write_text('x = 1\n')
    (m / 'src/simulation/gazebo_env_v4_backup.py').write_text('# old variant\n')
    (m / 'scripts/train_gazebo_ppo.py').write_text(
        "import os\n"
        "model_path = os.path.expanduser('~/Praxis/models/ppo_gazebo_v6')\n"
        "curves = os.path.expanduser('~/Praxis/data/gazebo_training_curves_v6.csv')\n")
    (m / 'scripts/evaluate_four_way_lidar.py').write_text(
        "import os\nimport pandas as pd\npath = 'ppo_lidar_guided'\n"
        "g = os.path.expanduser(f'~/Praxis/models/{path}')\n"
        "root = os.path.expanduser('~/Praxis')\n"
        "out = os.path.expanduser(\"~/Praxis/data\")\n"
        "with open(os.path.expanduser('~/Praxis/models/bc_lidar_policy.pkl'), 'rb') as f:\n    pass\n"
        "pd.DataFrame().to_csv(f'{out}/four_way_lidar_results.csv', index=False)\n"
        "bad = os.path.expanduser('~/Praxis/data/' + 'x')\n")
    (m / 'scripts/tabbed_script.py').write_text('def f():\n\treturn 1\n')
    (m / 'run_full_pipeline.py').write_text("print('root')\n")
    (m / 'worlds/praxis_multiagent_lidar.sdf').write_text(
        '<sdf><include><uri>/home/user/Praxis/assets/ycb/model</uri></include></sdf>\n')
    (m / 'assets/sku110k/density_params.json').write_text('{"p90": 200}\n')
    (m / 'models/ppo_lidar_guided.zip').write_bytes(os.urandom(512))
    (m / 'data/four_way_lidar_results.csv').write_text('system,completed\n')
    with open(m / 'data/dwb_lidar_demonstrations.csv', 'wb') as f:
        f.truncate(30 * 1024 * 1024)
    return m


def corrected_world(root):
    robot = ('      <link name="base_link">\n' + BASE_FIXED + '\n      </link>\n' +
             ('      <link name="wheel">\n        ' + MASS_FIXED + '\n' +
              WHEEL_FIXED + '\n      </link>\n') * 2)
    sdf = ('<?xml version="1.0" ?>\n<sdf version="1.9"><world name="praxis_warehouse">\n'
           + robot * 2 + '</world></sdf>\n')
    p = root / 'corrected_world.sdf'
    p.write_text(sdf)
    return p


def synthetic_data(root):
    d = root / 'synth' / 'data'
    d.mkdir(parents=True, exist_ok=True)

    rows = []
    spec = {'PPO_guided': (236, 146, 418, -98.3088), 'PPO_unguided': (450, 299, 51, -90.9731),
            'BC_ml': (395, 319, 86, -104.3025), 'Nav2_DWB': (363, 329, 108, -108.9568)}
    for s, (c, k, t, rw) in spec.items():
        for i in range(c + k + t):
            rows.append({'system': s, 'reward': rw, 'completed': i < c,
                         'collided': c <= i < c + k, 'timed_out': i >= c + k})
    pd.DataFrame(rows).to_csv(d / 'four_way_lidar_results.csv', index=False)

    rows = []
    for sig, cp, cd in [(0.0, 143, 131), (0.05, 152, 140), (0.15, 145, 129), (0.30, 146, 126)]:
        for s, c in [('PPO_gazebo', cp), ('Nav2_DWB', cd)]:
            rows += [{'system': s, 'noise_sigma': sig, 'completed': i < c,
                      'reward': 0.0, 'efficiency': 0.0, 'cycle_time': 100} for i in range(250)]
    pd.DataFrame(rows).to_csv(d / 'gazebo_trained_results_v4.csv', index=False)

    rows = []
    for s, c, rw in [('PPO_guided', 925, 86.8237), ('PPO_unguided', 923, 85.2462),
                     ('BC_ml', 848, 76.6341), ('Nav2_DWB', 890, 86.2487)]:
        rows += [{'system': s, 'noise_sigma': [0.0, 0.05, 0.15, 0.30][i % 4], 'reward': rw,
                  'efficiency': 0.0, 'completed': i < c} for i in range(1000)]
    pd.DataFrame(rows).to_csv(d / 'four_way_results.csv', index=False)

    def binary(name, a, na, ca, b, nb, cb):
        pd.DataFrame([{'system': a, 'completed': i < ca} for i in range(na)] +
                     [{'system': b, 'completed': i < cb} for i in range(nb)]
                     ).to_csv(d / name, index=False)
    binary('h3_test_b_ood.csv', 'guided', 500, 460, 'unguided', 500, 473)
    binary('h3_roadmap_results.csv', 'roadmap_followed', 1000, 910, 'roadmap_skipped', 1000, 912)

    rows = []
    for tier, g, u in [('low_load', 527, 545), ('medium', 386, 383),
                       ('high', 277, 279), ('congested', 221, 216)]:
        rows += [{'tier': tier, 'system': 'PPO_guided', 'completed': i < g} for i in range(600)]
        rows += [{'tier': tier, 'system': 'PPO_unguided', 'completed': i < u} for i in range(600)]
    pd.DataFrame(rows).to_csv(d / 'h3_stress_tiers.csv', index=False)
    return d


if __name__ == '__main__':
    target = pathlib.Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)
    mock_tree(target)
    corrected_world(target)
    synthetic_data(target)
    print(f'fixtures written to {target}')
