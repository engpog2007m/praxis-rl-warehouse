import os, threading, time
from . import executors, node
_WORLD = {'pos':[8.0,18.0], 'vx':0.0, 'run':False, 'subs':[]}
def init(): 
    _WORLD['run']=True
    threading.Thread(target=_sim, daemon=True).start()
def shutdown(): _WORLD['run']=False
def _sim():
    from tf2_msgs.msg import TFMessage
    from sensor_msgs.msg import LaserScan
    moves = os.environ.get('FAKE_MOVES','1')=='1'
    t=0
    while _WORLD['run']:
        if moves: _WORLD['pos'][0] += _WORLD['vx']*0.02
        for typ, cb in list(_WORLD['subs']):
            if typ is TFMessage: cb(TFMessage([tuple(_WORLD['pos']),(4.0,2.0)]))
            elif typ is LaserScan and t%3==0 and os.environ.get('FAKE_SCAN','1')=='1': cb(LaserScan())
        t+=1; time.sleep(0.02)
