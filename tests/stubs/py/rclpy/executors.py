import time
class SingleThreadedExecutor:
    def __init__(self): self._run=True
    def add_node(self,n): pass
    def spin(self):
        while self._run: time.sleep(0.05)
    def shutdown(self): self._run=False
