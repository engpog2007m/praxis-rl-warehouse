import sys
print("SCRIPT STARTING", flush=True)
sys.stdout.flush()
import sys, os, time, math, subprocess
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from simulation.data_registry import DataRegistry

TASK_IDS = [1, 2, 3, 4, 1001]
EPISODES = 15
MAX_TICKS = 100
TICK_HZ = 3.0
# Goals in ODOM frame (relative to spawn at world 2,2)
GOALS = {
    1: np.array([3.0, 3.0]),
    2: np.array([4.0, 2.0]),
    3: np.array([2.5, 3.5]),
    4: np.array([2.0, 3.0]),
    1001: np.array([3.5, 3.5]),
}

def get_odom(robot):
    try:
        r = subprocess.run(
            ['ros2', 'topic', 'echo', f'/{robot}/odom',
             '--field', 'pose.pose.position', '--once'],
            capture_output=True, text=True, timeout=2.0)
        x = y = None
        for line in r.stdout.split('\n'):
            l = line.strip()
            if l.startswith('x:'):
                x = float(l.split(':')[1].strip())
            if l.startswith('y:'):
                y = float(l.split(':')[1].strip())
        if x is not None and y is not None:
            return np.array([x, y])
    except Exception:
        pass
    return None

def get_pos(robot, default):
    r = get_odom(robot)
    if r is not None:
        return r
    return default

def send_vel(robot, vx, vy):
    msg = (f"{{linear: {{x: {vx:.3f}, y: {vy:.3f}, z: 0.0}}, "
           f"angular: {{x: 0.0, y: 0.0, z: 0.0}}}}")
    subprocess.Popen(
        ['ros2', 'topic', 'pub', '--once',
         f'/{robot}/cmd_vel',
         'geometry_msgs/msg/Twist', msg],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

class DWB:
    def __init__(self):
        self.rng = np.random.default_rng(42)
    def act(self, pos, goal):
        d = goal - pos
        n = max(np.linalg.norm(d), 1e-6)
        s = min(0.25 * n / 36.6, 0.2)
        e = self.rng.normal(0, 0.01)
        return (float(np.clip(s*d[0]/n+e, -0.2, 0.2)),
                float(np.clip(s*d[1]/n+e, -0.2, 0.2)))

print("Loading PPO...")
ppo = PPO.load(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models/ppo_warehouse'), device='cpu')
print("PPO loaded")
reg = DataRegistry()
dwb = DWB()

print("Testing odom...")
p1_init = get_pos('amr_1', np.array([2.0, 2.0]))
p2_init = get_pos('amr_2', np.array([4.0, 2.0]))
print(f"AMR_1={p1_init}  AMR_2={p2_init}")

records = []
dt = 1.0 / TICK_HZ

for tid in TASK_IDS:
    task = reg.get_task(tid)
    goal = GOALS[tid]
    for sigma in reg.noise_sigmas:
        for mass in [1.0, 1.5, 2.0]:
            for ep in range(EPISODES):
                p1 = get_pos('amr_1', p1_init).copy()
                p2 = get_pos('amr_2', p2_init).copy()
                # Goal alternates direction each episode to stay in bounds
                direction = 1 if ep % 2 == 0 else -1
                goal = p1 + GOALS[tid] * direction
                R1 = R2 = v1 = ticks = 0
                done1 = done2 = inter = False
                pd1 = pd2 = None

                for tick in range(MAX_TICKS):
                    p1 = get_pos('amr_1', p1)
                    p2 = get_pos('amr_2', p2)
                    d1 = float(np.linalg.norm(goal - p1))
                    d2 = float(np.linalg.norm(goal - p2))

                    n = (np.random.normal(0, sigma, 7).astype(np.float32)
                         if sigma > 0 else np.zeros(7, np.float32))
                    obs = np.clip(
                        np.array([p1[0], p1[1], goal[0], goal[1],
                                  d1, 0.0, 0.0], np.float32) + n,
                        [0, 0, 0, 0, 0, -1, -1],
                        [36.6, 36.0, 36.6, 36.0, 51.8, 1, 1])

                    a, _ = ppo.predict(obs, deterministic=True)
                    vx1 = float(a[0]) * 0.2
                    vy1 = float(a[1]) * 0.2
                    if task['is_heavy'] and mass > 1.25:
                        vx1 *= 0.75
                        vy1 *= 0.75
                    vx2, vy2 = dwb.act(p2, goal)

                    if not done1:
                        send_vel('amr_1', vx1, vy1)
                    if not done2:
                        send_vel('amr_2', vx2, vy2)

                    s1 = math.sqrt(vx1**2 + vy1**2)
                    pr1 = (pd1 - d1) * 10 if pd1 else 0
                    pr2 = (pd2 - d2) * 10 if pd2 else 0
                    ck = -0.5 * s1 if task['is_heavy'] else 0
                    cs = -2.0 if task['is_fragile'] and s1 > 0.15 else 0
                    R1 += pr1 + ck + cs - 0.1
                    R2 += pr2 - 0.1
                    if ck < 0 or cs < 0:
                        v1 += 1
                    pd1 = d1
                    pd2 = d2
                    ticks += 1

                    sep = float(np.linalg.norm(p1 - p2))
                    if sep < 0.4:
                        inter = True
                    if d1 < 2.0 and not done1:
                        done1 = True
                        R1 += 100
                    if d2 < 2.0 and not done2:
                        done2 = True
                        R2 += 100
                    if done1 and done2:
                        break
                    time.sleep(dt)

                send_vel('amr_1', 0, 0)
                send_vel('amr_2', 0, 0)
                time.sleep(0.5)

                for sn, R, done, vr in [
                    ('PPO_fleet', R1, done1, v1),
                    ('Nav2_DWB_fleet', R2, done2, 0)
                ]:
                    records.append({
                        'system': sn, 'task_id': tid,
                        'noise_sigma': sigma, 'mass_multiplier': mass,
                        'episode': ep, 'reward': round(float(R), 4),
                        'cycle_time': ticks,
                        'throughput': 1.0 if done and ticks < 80 else 0.0,
                        'efficiency': round(float(R)/ticks, 4) if ticks else 0,
                        'violation_rate': round(v1/ticks, 4) if ticks else 0,
                        'completed': done,
                        'inter_agent_collision': inter})

                if ep % 5 == 0:
                    print(f"task={tid} s={sigma:.2f} m={mass:.2f} "
                          f"ep={ep:02d} | "
                          f"PPO={R1:6.1f} d={done1} | "
                          f"DWB={R2:6.1f} d={done2} | "
                          f"col={inter}")

send_vel('amr_1', 0, 0)
send_vel('amr_2', 0, 0)

out = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data')
os.makedirs(out, exist_ok=True)
df = pd.DataFrame(records)
df.to_csv(f'{out}/gazebo_multiagent_results.csv', index=False)

print(f"\n{'='*55}")
print("MULTI-AGENT GAZEBO FLEET RESULTS")
print(f"{'='*55}")
print(f"Total records: {len(df):,}")
for sn in ['PPO_fleet', 'Nav2_DWB_fleet']:
    s = df[df['system'] == sn]
    print(f"\n{sn}:")
    print(f"  Throughput     : {s['throughput'].mean():.1%}")
    print(f"  Efficiency     : {s['efficiency'].mean():.4f}")
    print(f"  Violation rate : {s['violation_rate'].mean():.4f}")
    print(f"  Cycle time     : {s['cycle_time'].mean():.1f} ticks")
    print(f"  Inter-collision: {s['inter_agent_collision'].mean():.1%}")
    print(f"  Completed      : {s['completed'].mean():.1%}")
