class _T:
    def __init__(self,x,y):
        class tr: pass
        self.transform=tr(); self.transform.translation=type('t',(),{})()
        self.transform.translation.x=x; self.transform.translation.y=y
class TFMessage:
    def __init__(self, poses): self.transforms=[_T(x,y) for x,y in poses]
