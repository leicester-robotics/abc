import tempfile
import unittest
from pathlib import Path

class DriverSetupTests(unittest.TestCase):
    def test_prepared_sdk_has_verified_lifecycle_without_modifying_installed_sdk(self):
        from deploy.dashboard.driver_setup import prepare_sdk, verify_sdk
        import importlib.util
        installed=Path(importlib.util.find_spec('i2rt').origin).parent
        before=(installed/'motor_drivers/dm_driver.py').read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            target=prepare_sdk(Path(folder)/'sdk')
            verify_sdk(target)
            source=(target/'i2rt/motor_drivers/dm_driver.py').read_text()
            self.assertIn('missing disabled-status acknowledgement',source)
            self.assertIn('active_for_channel',source)
            (target/'i2rt/motor_drivers/dm_driver.py').write_text(source+'\n# changed\n')
            with self.assertRaisesRegex(RuntimeError,'modified'):verify_sdk(target)
        self.assertEqual((installed/'motor_drivers/dm_driver.py').read_bytes(),before)

    def test_disable_retry_reopens_transport_without_enabling(self):
        from deploy.dashboard.driver_setup import prepare_sdk
        import subprocess,sys,os
        with tempfile.TemporaryDirectory() as folder:
            root=prepare_sdk(Path(folder)/'sdk')
            code="""
from i2rt.motor_drivers.dm_driver import DMChainCanInterface
from unittest.mock import Mock,patch
import threading
c=DMChainCanInterface.__new__(DMChainCanInterface)
c.channel='fake';c.motor_list=[(i,'DM4310') for i in range(1,8)]
c._close_lock=threading.Lock();c._closed=False;c._close_error=None
c._transport_closed=False;c._control_thread=None;c._interface_kwargs={}
c._disabled_motors=[];c._disable_failures={};c.running=True
c.motor_interface=Mock();c.motor_interface.try_receive_message.return_value=None
c.motor_interface.motor_off.side_effect=RuntimeError('lost ACK')
c._active_chains[c.channel]=c
try:c.close()
except RuntimeError:pass
else:raise AssertionError('missing shutdown fault')
assert c.active_for_channel('fake') is c
fresh=Mock();fresh.try_receive_message.return_value=None
with patch('i2rt.motor_drivers.dm_driver.DMSingleMotorCanInterface',return_value=fresh):c.close()
assert c._closed and c.active_for_channel('fake') is None
assert c._disabled_motors==list(range(1,8))
fresh.motor_on.assert_not_called()
"""
            result=subprocess.run([sys.executable,'-c',code],env={**os.environ,'PYTHONPATH':str(root)},capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_enable_drains_fault_clear_replies_before_waiting_for_enable_ack(self):
        from deploy.dashboard.driver_setup import prepare_sdk
        import subprocess,sys,os
        with tempfile.TemporaryDirectory() as folder:
            root=prepare_sdk(Path(folder)/'sdk')
            code="""
from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface
from types import SimpleNamespace
from collections import deque
q=deque();events=[]
c=DMSingleMotorCanInterface.__new__(DMSingleMotorCanInterface)
c.parse_recv_message=lambda msg,*args,**kw: msg
c._get_frame_id=lambda i:i
c.try_receive_message=lambda *args,**kwargs: q.popleft() if q else None
first=[True]
def send(*args):
 if first[0]:first[0]=False;return SimpleNamespace(error_code='0xd',error_message='loss communication')
 if q:return q.popleft()
 events.append('fresh-enabled');return SimpleNamespace(error_code='0x1',error_message='normal')
def clear(*args,**kw):
 for _ in range(3):q.append(SimpleNamespace(error_code='0x0',error_message='disabled'))
c._send_message_get_response=send;c.clean_error=clear
result=c.motor_on(1,'DM4310')
assert result.error_code=='0x1',result
assert events==['fresh-enabled']
"""
            result=subprocess.run([sys.executable,'-c',code],env={**os.environ,'PYTHONPATH':str(root)},capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
