"""Real-browser tests. Enable with DASHBOARD_BROWSER_TESTS=1 and Playwright installed."""
import os
import queue
import socket
import time
import unittest


@unittest.skipUnless(os.getenv('DASHBOARD_BROWSER_TESTS') == '1', 'opt-in browser integration')
class BrowserTests(unittest.TestCase):
    def test_client_heartbeat_and_disconnect(self):
        import viser
        from playwright.sync_api import sync_playwright
        from deploy.dashboard.config import DashboardConfig
        from deploy.dashboard.control import ControlSupervisor
        from deploy.dashboard.ui import DashboardUI
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        server=viser.ViserServer(host='127.0.0.1',port=port,verbose=False)
        supervisor=ControlSupervisor([],{},lambda _:self.fail('physical factory called'))
        ui=DashboardUI(server,supervisor,DashboardConfig(),queue.Queue(),demo=True)
        try:
            with sync_playwright() as p:
                browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
                page=browser.new_page();page.goto(f'http://127.0.0.1:{port}')
                page.get_by_role('button',name='Connect robot',exact=True).wait_for(timeout=30000)
                deadline=time.monotonic()+10
                while not supervisor._heartbeats and time.monotonic()<deadline:page.wait_for_timeout(100)
                self.assertTrue(supervisor._heartbeats,'Browser did not send application heartbeat')
                client=next(iter(supervisor._heartbeats)); first=supervisor._heartbeats[client]
                page.wait_for_timeout(350)
                self.assertGreater(supervisor._heartbeats[client],first)
                # Hiding the page stops our application timer (not the websocket).
                page.evaluate("Object.defineProperty(document,'visibilityState',{get:()=> 'hidden', configurable:true})")
                page.wait_for_timeout(650)
                self.assertFalse(supervisor._lease_valid(client,time.monotonic()))
                second=browser.new_page();second.goto(f'http://127.0.0.1:{port}')
                second.get_by_role('button',name='Connect robot',exact=True).wait_for(timeout=30000)
                second.wait_for_timeout(350)
                self.assertGreaterEqual(len(supervisor._heartbeats),2)
                self.assertFalse(supervisor._lease_valid(client,time.monotonic()))
                page.close();second.close();browser.close()
                deadline=time.monotonic()+2
                while supervisor._heartbeats and time.monotonic()<deadline:time.sleep(.01)
                self.assertFalse(supervisor._heartbeats)
        finally:server.stop()
