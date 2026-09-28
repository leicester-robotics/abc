import unittest
import numpy as np
from deploy.dashboard.config import ArmConfig
from deploy.dashboard.samples import LatestSample
from deploy.dashboard.gello import GelloReader, map_encoder, encoder_radians, calibration_offsets

class Connection:
    def __init__(self): self.closed=False
    def disconnect(self): self.closed=True

class Sync:
    def __init__(self): self.missing=None; self.result=0
    def addParam(self, i): return True
    def txRxPacket(self): return self.result
    def isAvailable(self, i, a, n): return i != self.missing
    def getData(self, i, a, n): return 2048

class GelloTests(unittest.TestCase):
    def setUp(self):
        self.cfg=ArmConfig('left','/dev/test',list(range(7)),[1]*7, [0.]*7)
    def test_encoder_and_gripper_mapping(self):
        np.testing.assert_equal(encoder_radians(np.full(7,2048)),0.)
        for angle,expected in zip(self.cfg.gripper_range,[0.,1.]):
            counts=np.full(7,2048.,dtype=float); counts[-1]=(angle/np.pi+1)*2048
            self.assertAlmostEqual(map_encoder(counts,self.cfg)[-1],expected)
        self.cfg.signs[0]=-1; self.cfg.offsets[0]=.25
        self.assertAlmostEqual(map_encoder(np.full(7,2048),self.cfg)[0],.25)
    def test_calibration_must_be_explicit(self):
        self.cfg.offsets=None
        self.assertIsNone(map_encoder(np.full(7,2048),self.cfg))
        offsets=calibration_offsets(np.full(7,2048),self.cfg.signs)
        self.assertEqual(len(offsets),7)
    def test_missing_servo_and_failed_reads_preserve_sample_age(self):
        sync=Sync(); conn=Connection(); buf=LatestSample()
        reader=GelloReader(self.cfg,buf,connection_factory=lambda:conn,sync_factory=lambda c,ids:sync)
        reader.open(); reader.read_once()
        before=buf.read().acquired_at
        sync.missing=3
        with self.assertRaises(ConnectionError): reader.read_once()
        self.assertEqual(buf.read().acquired_at,before)
        sync.missing=None; sync.result=1
        with self.assertRaises(ConnectionError): reader.read_once()
        reader.close(); self.assertTrue(conn.closed)
    def test_uncalibrated_sample_has_raw_only(self):
        self.cfg.offsets=None; buf=LatestSample()
        reader=GelloReader(self.cfg,buf,connection_factory=Connection,sync_factory=lambda c,ids:Sync())
        reader.open(); reader.read_once(); reader.close()
        self.assertIsNone(buf.read().position)
        np.testing.assert_equal(buf.read().counts,2048)
