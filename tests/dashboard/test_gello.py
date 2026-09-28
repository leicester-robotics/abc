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

    def test_sequential_fallback_reads_every_servo(self):
        class Packet:
            def read4ByteTxRx(self,port,motor,address):return 2048+motor,0,0
        conn=Connection();conn.packetHandler=Packet();conn.portHandler=object()
        sync=Sync();sync.result=1;buf=LatestSample()
        reader=GelloReader(self.cfg,buf,connection_factory=lambda:conn,sync_factory=lambda c,ids:sync)
        reader.open();reader.read_once();reader.close()
        np.testing.assert_equal(buf.read().counts,np.arange(7)+2048)
        self.assertEqual(buf.read().raw['read_mode'],'individual')

    def test_reconnect_retries_sync_after_transient_failure(self):
        class Packet:
            def read4ByteTxRx(self,port,motor,address):return 2048,0,0
        conn=Connection();conn.packetHandler=Packet();conn.portHandler=object()
        sync=Sync();sync.result=1;buf=LatestSample()
        reader=GelloReader(self.cfg,buf,connection_factory=lambda:conn,sync_factory=lambda c,ids:sync)
        reader.open();reader.read_once();reader._close_device()
        sync.result=0;reader.open();reader.read_once();reader.close()
        self.assertEqual(buf.read().raw['read_mode'],'sync')

    def test_saved_calibration_signs_home_and_encoder_wrap(self):
        from deploy.dashboard.saved_calibration import Calibration
        from pathlib import Path
        path='deploy/dashboard/calibrations/left.relative.json'
        cal=Calibration.load(Path(path))
        cfg=ArmConfig('left',cal.port,cal.ids,cal.signs+[1],calibration_path=path,baudrate=cal.baudrate)
        counts=np.rint(np.asarray(cal.home_raw)*2048/np.pi).astype(int)
        # Joint five's encoder can reboot one revolution lower.
        counts[4]-=4096
        class PoseSync(Sync):
            def getData(self,i,a,n):return int(counts[i-1]) % 2**32
        buf=LatestSample();reader=GelloReader(cfg,buf,connection_factory=Connection,sync_factory=lambda c,ids:PoseSync())
        reader.open();reader.read_once()
        np.testing.assert_allclose(buf.read().position[:6],cal.home_target,atol=1e-8)
        counts[2]+=100;counts[3]+=100
        reader.read_once()
        self.assertGreater(buf.read().position[2],.15)
        self.assertGreater(buf.read().position[3],1.72)
        self.assertTrue(buf.read().calibrated)
        reader.close()

    def test_saved_calibration_follows_device_when_side_label_is_corrected(self):
        from deploy.dashboard.saved_calibration import Calibration
        from pathlib import Path
        path='deploy/dashboard/calibrations/left.relative.json'
        cal=Calibration.load(Path(path))
        cfg=ArmConfig('right',cal.port,cal.ids,cal.signs+[1],calibration_path=path,baudrate=cal.baudrate)
        reader=GelloReader(cfg,LatestSample(),connection_factory=Connection,sync_factory=lambda c,ids:Sync())
        reader.open();reader.read_once();reader.close()

    def test_haptics_expire_independently_of_position_reads(self):
        from deploy.dashboard.haptics import HapticTarget
        from unittest.mock import Mock,patch
        import time
        commands=LatestSample();output=Mock();output.active=False;output.original={};output.current=np.zeros(7)
        def enable():output.active=True
        def close():output.active=False
        output.enable.side_effect=enable;output.close.side_effect=close
        reader=GelloReader(self.cfg,LatestSample(),connection_factory=Connection,sync_factory=lambda c,ids:Sync(),haptics=commands)
        with patch('deploy.dashboard.gello.HapticOutput',return_value=output):
            reader.open();reader.read_once();output.enable.assert_not_called()
            commands.publish(HapticTarget(time.monotonic(),np.ones(7)))
            reader.read_once();self.assertTrue(output.active)
            commands.publish(HapticTarget(time.monotonic()-1.,np.ones(7)))
            reader.read_once();self.assertFalse(output.active)
            reader.close()

    def test_serial_port_closes_even_if_haptic_release_fails(self):
        from unittest.mock import Mock
        connection=Connection()
        reader=GelloReader(self.cfg,LatestSample(),connection_factory=lambda:connection,sync_factory=lambda c,ids:Sync())
        reader.open();reader.haptic_output=Mock()
        reader.haptic_output.close.side_effect=RuntimeError('off ACK lost')
        with self.assertRaisesRegex(RuntimeError,'off ACK lost'):reader._close_device()
        self.assertTrue(connection.closed)

    def test_reconnect_restores_pending_haptic_settings_before_enable(self):
        from unittest.mock import Mock
        from deploy.dashboard.haptics import HapticTarget
        import time
        commands=LatestSample();connections=[]
        def connect():
            c=Connection();connections.append(c);return c
        reader=GelloReader(self.cfg,LatestSample(),connection_factory=connect,sync_factory=lambda c,ids:Sync(),haptics=commands)
        reader.open();output=Mock();output.active=False;output.original={1:(3,100)}
        reader.haptic_output=output;output.close.side_effect=RuntimeError('off ACK lost')
        with self.assertRaises(RuntimeError):reader._close_device()
        reader.open()
        self.assertIs(output.connection,connections[-1])
        self.assertTrue(connections[0].closed)
        commands.publish(HapticTarget(time.monotonic(),np.ones(7)))
        with self.assertRaises(RuntimeError):reader._update_haptics()
        output.enable.assert_not_called()
        def restored():output.original.clear()
        output.close.side_effect=restored
        reader._update_haptics();output.enable.assert_called_once()
        reader.close()

    def test_missed_packet_retries_without_reopening_or_refreshing_sample(self):
        sync=Sync();buf=LatestSample();conn=Connection()
        reader=GelloReader(self.cfg,buf,connection_factory=lambda:conn,sync_factory=lambda c,ids:sync)
        reader.open();reader.read_once();before=buf.read().acquired_at
        sync.missing=3
        with self.assertRaises(TimeoutError):reader.read_once()
        self.assertFalse(conn.closed)
        self.assertEqual(buf.read().acquired_at,before)
        sync.missing=None;reader.read_once()
        self.assertGreater(buf.read().acquired_at,before)
        reader.close()

    def test_missing_encoder_batch_zeros_haptics_and_preserves_write_failure(self):
        from unittest.mock import Mock
        sync=Sync();reader=GelloReader(self.cfg,LatestSample(),connection_factory=Connection,sync_factory=lambda c,ids:sync)
        reader.open();output=Mock();output.active=True;reader.haptic_output=output
        sync.missing=3
        with self.assertRaises(TimeoutError):reader.read_once()
        np.testing.assert_array_equal(output.write.call_args.args[0],np.zeros(7))
        output.write.side_effect=ConnectionError('current write failed')
        with self.assertRaisesRegex(ConnectionError,'current write failed'):reader.read_once()
        reader.close()

    def test_worker_recovers_missed_packet_on_same_connection(self):
        import time
        sync=Sync();buf=LatestSample();connections=[]
        def connect():
            c=Connection();connections.append(c);return c
        reader=GelloReader(self.cfg,buf,connection_factory=connect,sync_factory=lambda c,ids:sync)
        reader.start()
        try:
            deadline=time.monotonic()+1
            while buf.read() is None and time.monotonic()<deadline:time.sleep(.005)
            self.assertIsNotNone(buf.read())
            sync.missing=3
            deadline=time.monotonic()+1
            while not buf.error and time.monotonic()<deadline:time.sleep(.005)
            self.assertTrue(buf.transient_error)
            sync.missing=None
            deadline=time.monotonic()+1
            while buf.error and time.monotonic()<deadline:time.sleep(.005)
            self.assertFalse(buf.error)
            self.assertEqual(len(connections),1)
        finally:reader.close()

    def test_stronger_haptics_still_ramp_current(self):
        from unittest.mock import Mock,patch
        from deploy.dashboard.haptics import HapticTarget
        commands=LatestSample();commands.publish(HapticTarget(1.1,np.full(7,75.)))
        reader=GelloReader(self.cfg,LatestSample(),haptics=commands)
        output=Mock();output.active=True;reader.haptic_output=output;reader._haptic_previous=1.
        with patch('deploy.dashboard.gello.time.monotonic',return_value=1.1):reader._update_haptics()
        np.testing.assert_allclose(output.write.call_args.args[0],np.full(7,15.))

    def test_hold_uses_encoder_pose_and_clears_on_stale_command(self):
        from unittest.mock import Mock,patch
        from deploy.dashboard.haptics import HapticTarget
        commands=LatestSample();reader=GelloReader(self.cfg,LatestSample(),haptics=commands)
        output=Mock();output.active=True;output.original={};reader.haptic_output=output
        reader._haptic_previous=1.
        commands.publish(HapticTarget(1.1,np.zeros(7),hold_percent=100.))
        with patch('deploy.dashboard.gello.time.monotonic',return_value=1.1):reader._update_haptics(np.zeros(7))
        commands.publish(HapticTarget(1.14,np.zeros(7),hold_percent=100.))
        with patch('deploy.dashboard.gello.time.monotonic',return_value=1.14):reader._update_haptics(np.full(7,.02))
        self.assertTrue(np.all(output.write.call_args.args[0][:6]<0))
        self.assertEqual(output.write.call_args.args[0][-1],0.)
        with patch('deploy.dashboard.gello.time.monotonic',return_value=1.5):reader._update_haptics(np.full(7,.02))
        self.assertIsNone(reader.pose_hold.anchor)
        output.close.assert_called_once()

    def test_slow_successful_batch_cannot_drive_hold_from_old_encoders(self):
        from unittest.mock import Mock,patch
        sync=Sync();reader=GelloReader(self.cfg,LatestSample(),connection_factory=Connection,sync_factory=lambda c,ids:sync)
        reader.open();output=Mock();output.active=True;reader.haptic_output=output
        with patch('deploy.dashboard.gello.time.monotonic',side_effect=[1.,1.101,1.101,1.101]):
            with self.assertRaises(TimeoutError):reader.read_once()
        np.testing.assert_array_equal(output.write.call_args.args[0],np.zeros(7))
        self.assertIsNone(reader.buffer.read())
        self.assertIsNone(reader.pose_hold.anchor)
        reader.close()
