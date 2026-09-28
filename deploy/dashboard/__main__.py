"""Run with python -m deploy.dashboard; no physical connection on startup."""
import argparse
from pathlib import Path
from .config import DashboardConfig, load_config, validate


def main():
    parser=argparse.ArgumentParser(description='GELLO/YAM dashboard: MuJoCo, cameras, raw data, explicit physical teleop.')
    parser.add_argument('--config',type=Path,help='Station JSON; requires explicit arm assignments')
    parser.add_argument('--port',type=int,help='Loopback HTTP/WebSocket port (default 8080)')
    parser.add_argument('--model-path',help='Bimanual YAM MuJoCo scene')
    parser.add_argument('--asset-dir',help='Directory containing the YAM STL meshes')
    parser.add_argument('--demo',action='store_true',help='Synthetic simulation only; no hardware opened')
    parser.add_argument('--discover',action='store_true',help='List camera and serial identities without moving motors')
    parser.add_argument('--status-path',type=Path,default=Path('outputs/dashboard/status.json'))
    args=parser.parse_args()
    if args.discover:
        import json
        from .gello import discover_devices
        print(json.dumps(discover_devices(),indent=2));return
    if not args.config and not args.demo:parser.error('Pass --config STATION.json, or --demo for hardware-free simulation')
    cfg=load_config(args.config) if args.config else DashboardConfig()
    if args.port is not None:cfg.port=args.port
    if args.model_path:cfg.model_path=args.model_path
    if args.asset_dir:cfg.asset_dir=args.asset_dir
    from .app import run
    raise SystemExit(run(validate(cfg),args.demo,args.config,args.status_path))


if __name__=='__main__':main()
