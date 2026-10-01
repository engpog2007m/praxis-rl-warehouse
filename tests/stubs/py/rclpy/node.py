class _Pub:
    def publish(self, m):
        from rclpy import _WORLD; _WORLD['vx'] = m.linear.x
    def get_subscription_count(self): return 1
class Node:
    def __init__(self, name): pass
    def create_publisher(self, t, topic, q): return _Pub()
    def create_subscription(self, t, topic, cb, q):
        from rclpy import _WORLD; _WORLD['subs'].append((t, cb))
    def get_topic_names_and_types(self):
        import os
        base=['/amr_1/cmd_vel','/amr_1/odom','/world/praxis_warehouse/dynamic_pose/info','/amr_1/scan','/amr_2/scan']
        if os.environ.get('FAKE_DROP_POSE')=='1': base.remove('/world/praxis_warehouse/dynamic_pose/info')
        return [(t,['x']) for t in base]
    def destroy_node(self): pass
