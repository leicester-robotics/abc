import socket
import unittest
from unittest.mock import patch
from deploy.dashboard.app import ensure_port_free, make_workers
from deploy.dashboard.config import DashboardConfig

class AppTests(unittest.TestCase):
    def test_occupied_port_is_error(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));sock.listen()
            with self.assertRaises(OSError):ensure_port_free(sock.getsockname()[1])
    def test_demo_never_opens_devices(self):
        with patch('deploy.dashboard.app.GelloReader') as gello, patch('deploy.dashboard.app.CameraWorker') as camera:
            self.assertEqual(make_workers(DashboardConfig(),{}, {}, {}, demo=True),[])
            gello.assert_not_called();camera.assert_not_called()
