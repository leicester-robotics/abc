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
                page.get_by_role('button',name='Reset simulation',exact=True).wait_for(timeout=30000)
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
                second.get_by_role('button',name='Reset simulation',exact=True).wait_for(timeout=30000)
                second.wait_for_timeout(350)
                self.assertGreaterEqual(len(supervisor._heartbeats),2)
                self.assertFalse(supervisor._lease_valid(client,time.monotonic()))
                page.close();second.close();browser.close()
                deadline=time.monotonic()+2
                while supervisor._heartbeats and time.monotonic()<deadline:time.sleep(.01)
                self.assertFalse(supervisor._heartbeats)
        finally:server.stop()

    def test_streaming_camera_panels_do_not_crash_react(self):
        import threading
        import numpy as np
        import viser
        from playwright.sync_api import sync_playwright
        from deploy.dashboard.config import DashboardConfig,CameraConfig
        from deploy.dashboard.control import ControlSupervisor
        from deploy.dashboard.samples import CameraSample,ArmSample
        from deploy.dashboard.ui import DashboardUI
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        server=viser.ViserServer(host='127.0.0.1',port=port,verbose=False)
        supervisor=ControlSupervisor([],{},lambda _:self.fail('physical factory called'))
        cfg=DashboardConfig(cameras=[CameraConfig(str(i),f'Camera {i}',True) for i in range(3)])
        ui=DashboardUI(server,supervisor,cfg,queue.Queue(),demo=True)
        stop=threading.Event()
        def update():
            rng=np.random.default_rng(0)
            while not stop.is_set():
                now=time.monotonic();rgb=rng.integers(0,255,(480,640,3),dtype=np.uint8)
                ui.update(dict(control=supervisor.status(),leaders={},physical={},
                    simulation={s:ArmSample(now,np.zeros(7)) for s in ('left','right')},errors={},camera_errors={},
                    cameras={str(i):CameraSample(now,rgb,rgb,'enabled') for i in range(3)},sim_status='Test'))
                stop.wait(1/15)
        thread=threading.Thread(target=update);thread.start()
        try:
            with sync_playwright() as p:
                browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
                page=browser.new_page(viewport={'width':1500,'height':1050});errors=[]
                page.on('pageerror',lambda error:errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{port}')
                page.get_by_role('button',name='Reset simulation',exact=True).wait_for(timeout=30000)
                page.get_by_role('tab',name='Cameras',exact=True).click()
                page.wait_for_timeout(3000)
                self.assertFalse(errors,errors)
                self.assertGreaterEqual(page.locator('img').count(),6)
                browser.close()
        finally:
            stop.set();thread.join();server.stop()

    def test_dashboard_has_separate_tabs(self):
        import viser
        from playwright.sync_api import sync_playwright
        from deploy.dashboard.config import DashboardConfig
        from deploy.dashboard.control import ControlSupervisor
        from deploy.dashboard.ui import DashboardUI
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        server=viser.ViserServer(host='127.0.0.1',port=port,verbose=False)
        supervisor=ControlSupervisor([],{},lambda _:None)
        ui=DashboardUI(server,supervisor,DashboardConfig(),queue.Queue(),demo=True)
        try:
            with sync_playwright() as p:
                browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
                page=browser.new_page();page.goto(f'http://127.0.0.1:{port}')
                page.get_by_role('button',name='Reset simulation',exact=True).wait_for(timeout=30000)
                self.assertEqual(page.get_by_role('tab').all_text_contents(),['Simulation','Cameras','Arm data','Setup'])
                page.get_by_role('tab',name='Arm data',exact=True).click()
                self.assertFalse(page.get_by_role('button',name='Reset simulation',exact=True).is_visible())
                page.get_by_role('tab',name='GELLO',exact=True).wait_for()
                page.get_by_role('tab',name='Physical YAM',exact=True).click()
                page.get_by_role('tab',name='Right',exact=True).click()
                self.assertTrue(page.locator('p:visible').filter(has_text='Physical YAM: unavailable').is_visible())
                page.get_by_role('tab',name='Cameras',exact=True).click()
                page.wait_for_timeout(400)
                self.assertIn('Cameras',ui.client_views.values())
                browser.close()
        finally:server.stop()
