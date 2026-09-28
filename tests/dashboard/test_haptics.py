import unittest
from types import SimpleNamespace
import numpy as np
from deploy.dashboard.samples import ArmSample

class HapticTests(unittest.TestCase):
    def test_reflection_sign_cap_and_stale_shutoff(self):
        from deploy.dashboard.haptics import reflected_current
        arm=SimpleNamespace(signs=[1,-1,1,1,1,1,1])
        sample=ArmSample(10.,np.zeros(7),effort=np.array([10.,10.,-10.,0.,0.,0.,2.]),raw={'gripper_direction':-1})
        current=reflected_current(arm,sample,10.,strength=1.)
        np.testing.assert_equal(current,[-25,25,25,0,0,0,10])
        self.assertIsNone(reflected_current(arm,sample,10.2,strength=1.))

    def test_enable_current_zero_then_torque_and_shutdown_restores_configuration(self):
        from deploy.dashboard.haptics import HapticOutput
        class Packet:
            def __init__(self):self.reg={11:3,38:1750,64:0,98:0,102:0};self.writes=[]
            def ping(self,p,i):return 1200,0,0
            def read1ByteTxRx(self,p,i,a):return self.reg[a],0,0
            read2ByteTxRx=read1ByteTxRx
            def write1ByteTxRx(self,p,i,a,v):self.reg[a]=v;self.writes.append((a,v));return 0,0
            write2ByteTxRx=write1ByteTxRx
        packet=Packet();conn=SimpleNamespace(packetHandler=packet,portHandler=object())
        out=HapticOutput(conn,[1]);out.enable()
        self.assertEqual(packet.reg[38],25)
        self.assertEqual(packet.reg[98],5)
        self.assertLess(packet.writes.index((102,0)),packet.writes.index((64,1)))
        out.write(np.array([100.]));self.assertEqual(packet.reg[102],25)
        out.close()
        self.assertEqual(packet.reg[64],0)
        self.assertEqual(packet.reg[102],0)
        self.assertEqual(packet.reg[11],3)
        self.assertEqual(packet.reg[38],1750)
        self.assertEqual(packet.reg[98],0)
