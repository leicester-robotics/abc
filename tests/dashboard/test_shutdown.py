import unittest
from deploy.dashboard.app import shutdown_resources

class ShutdownTests(unittest.TestCase):
    def test_all_resources_attempted_after_one_failure(self):
        calls=[]
        class Resource:
            def __init__(self,name,fail=False):self.name=name;self.fail=fail
            def close(self):
                calls.append(self.name)
                if self.fail:raise RuntimeError('device cleanup failed')
        class Server:
            def stop(self):calls.append('server')
        with self.assertRaisesRegex(RuntimeError,'cleanup failed'):
            shutdown_resources(Resource('supervisor',True),[Resource('camera'),Resource('serial')],Server())
        self.assertEqual(calls,['supervisor','camera','serial','server'])
