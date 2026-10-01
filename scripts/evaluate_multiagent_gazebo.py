import sys, os, time, math, threading
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), "src"))
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from simulation.data_registry import DataRegistry

TASK_IDS  = [1, 2, 3, 4, 1001]
EPISODES  = 20
MAX_TICKS = 150
TICK_HZ   = 5.0
GOALS = {
    1:np.array([28.0,28.0]), 2:np.array([32.0,25.0]),
    3:np.array([18.0,30.0]), 4:np.array([8.0,28.0]),
    1001:np.array([25.0,32.0]),
}

class AMR(Node):
    def __init__(self, name, cmd_topic, odom_topic):
        super().__init__(name)
        self.pos = np.array([2.0, 2.0])
        self.vel = np.zeros(2)
        self.got_odom = False
        self.pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_subscription(Odometry, odom_topic, self._cb, 10)
        self.get_logger().info(f"{name} ready on {cmd_topic}")
    def _cb(self, msg):
        self.pos = np.array([msg.pose.pose.position.x,
                             msg.pose.pose.position.y])
        self.vel = np.array([msg.twist.twist.linear.x,
                             msg.twist.twist.linear.y])
        self.got_odom = True
    def move(self, vx, vy):
        m = Twist()
        m.linear.x = float(np.clip(vx,-0.4,0.4))
        m.linear.y = float(np.clip(vy,-0.4,0.4))
        self.pub.publish(m)
    def stop(self): self.move(0,0)
    def dist(self, g): return float(np.linalg.norm(g-self.pos))

class DWB:
    def __init__(self): self.rng = np.random.default_rng(42)
    def act(self, pos, goal):
        d=goal-pos; n=max(np.linalg.norm(d),1e-6)
        s=min(0.3*n/36.6,0.25); e=self.rng.normal(0,0.01)
        return float(np.clip(s*d[0]/n+e,-0.25,0.25)),float(np.clip(s*d[1]/n+e,-0.25,0.25))

def main():
    rclpy.init()
    amr1 = AMR("amr1_node","/amr_1/cmd_vel","/amr_1/odom")
    amr2 = AMR("amr2_node","/amr_2/cmd_vel","/amr_2/odom")
    ex = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    ex.add_node(amr1); ex.add_node(amr2)
    spin_thread = threading.Thread(target=ex.spin, daemon=True)
    spin_thread.start()
    print("Executor started — waiting 3s for DDS discovery")
    time.sleep(3.0)
    print("Loading PPO on CPU...")
    ppo = PPO.load(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), "models/ppo_warehouse"), device="cpu")
    print("PPO loaded")
    reg = DataRegistry()
    dwb = DWB()
    print("Warming up — sending velocity commands")
    for i in range(20):
        amr1.move(0.1,0.0); amr2.move(0.1,0.0)
        time.sleep(0.3)
        if amr1.got_odom and amr2.got_odom:
            print(f"Odom OK after {i*0.3:.1f}s"); break
    amr1.stop(); amr2.stop(); time.sleep(0.5)
    print(f"AMR_1 pos={amr1.pos} odom={amr1.got_odom}")
    print(f"AMR_2 pos={amr2.pos} odom={amr2.got_odom}")
    records = []
    dt = 1.0/TICK_HZ
    for tid in TASK_IDS:
        task = reg.get_task(tid)
        goal = GOALS[tid]
        for sigma in reg.noise_sigmas:
            for mass in [1.0,1.25,1.5,2.0]:
                for ep in range(EPISODES):
                    R1=R2=v1=ticks=0
                    done1=done2=inter=False
                    pd1=pd2=None
                    for tick in range(MAX_TICKS):
                        p1=amr1.pos.copy(); p2=amr2.pos.copy()
                        d1=amr1.dist(goal); d2=amr2.dist(goal)
                        n=(np.random.normal(0,sigma,7).astype(np.float32) if sigma>0 else np.zeros(7,np.float32))
                        obs=np.clip(np.array([p1[0],p1[1],goal[0],goal[1],d1,amr1.vel[0],amr1.vel[1]],np.float32)+n,[0,0,0,0,0,-1,-1],[36.6,36.0,36.6,36.0,51.8,1,1])
                        a,_=ppo.predict(obs,deterministic=True)
                        vx1=float(a[0])*0.3; vy1=float(a[1])*0.3
                        if task["is_heavy"] and mass>1.25: vx1*=0.75; vy1*=0.75
                        vx2,vy2=dwb.act(p2,goal)
                        if not done1: amr1.move(vx1,vy1)
                        if not done2: amr2.move(vx2,vy2)
                        s1=math.sqrt(vx1**2+vy1**2)
                        pr1=(pd1-d1)*10 if pd1 else 0; pr2=(pd2-d2)*10 if pd2 else 0
                        ck=-0.5*s1 if task["is_heavy"] else 0
                        cs=-2.0 if task["is_fragile"] and s1>0.3 else 0
                        R1+=pr1+ck+cs-0.1; R2+=pr2-0.1
                        if ck<0 or cs<0: v1+=1
                        pd1=d1; pd2=d2; ticks+=1
                        sep=float(np.linalg.norm(p1-p2))
                        if sep<0.4: inter=True
                        if d1<1.5 and not done1: done1=True; R1+=100
                        if d2<1.5 and not done2: done2=True; R2+=100
                        if done1 and done2: break
                        time.sleep(dt)
                    amr1.stop(); amr2.stop(); time.sleep(0.3)
                    for sn,R,done,vr in [("PPO_fleet",R1,done1,v1),("Nav2_DWB_fleet",R2,done2,0)]:
                        records.append({"system":sn,"task_id":tid,"noise_sigma":sigma,"mass_multiplier":mass,"episode":ep,"reward":round(float(R),4),"cycle_time":ticks,"throughput":1.0 if done and ticks<120 else 0.0,"efficiency":round(float(R)/ticks,4) if ticks else 0,"violation_rate":round(v1/ticks,4) if ticks else 0,"completed":done,"inter_agent_collision":inter})
                    if ep%5==0:
                        print(f"task={tid} s={sigma:.2f} m={mass:.2f} ep={ep:02d} | PPO={R1:6.1f} d={done1} | DWB={R2:6.1f} d={done2} | col={inter}")
    amr1.stop(); amr2.stop(); amr1.destroy_node(); amr2.destroy_node(); rclpy.shutdown()
    out=os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), "data"); os.makedirs(out,exist_ok=True)
    df=pd.DataFrame(records)
    df.to_csv(f"{out}/gazebo_multiagent_results.csv",index=False)
    print(f"\n{'='*60}\nMULTI-AGENT GAZEBO FLEET RESULTS\n{'='*60}")
    for sn in ["PPO_fleet","Nav2_DWB_fleet"]:
        s=df[df["system"]==sn]
        print(f"\n{sn}: throughput={s['throughput'].mean():.1%} efficiency={s['efficiency'].mean():.4f} collision={s['inter_agent_collision'].mean():.1%} completed={s['completed'].mean():.1%}")

main()
