"""Prepare an isolated SDK copy; never alter the station's shared installation."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import subprocess

PATCH=Path(__file__).with_name('patches')/'i2rt-lifecycle.patch'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_sdk(target):
    target=Path(target).resolve()
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite SDK directory: {target}')
    source=Path(importlib.util.find_spec('i2rt').origin).parent
    target.mkdir(parents=True)
    try:
        shutil.copytree(source,target/'i2rt',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        subprocess.run(['patch','--batch','--fuzz=0','-p1','-d',str(target),'-i',str(PATCH.resolve())],check=True,capture_output=True)
        files={str(p.relative_to(target)):digest(p) for p in (target/'i2rt').rglob('*.py')}
        (target/'dashboard-sdk.json').write_text(json.dumps({'patch':digest(PATCH),'files':files},indent=2))
        verify_sdk(target)
        return target
    except BaseException:
        shutil.rmtree(target)
        raise


def verify_sdk(target):
    target=Path(target).resolve()
    try:
        manifest=json.loads((target/'dashboard-sdk.json').read_text())
        if manifest['patch']!=digest(PATCH):raise ValueError('patch changed')
        for relative,expected in manifest['files'].items():
            if digest(target/relative)!=expected:raise ValueError(relative)
    except (OSError,ValueError,KeyError) as error:
        raise RuntimeError(f'Dashboard SDK missing or modified: {error}') from error
    return target


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target',type=Path)
    args=parser.parse_args()
    print(prepare_sdk(args.target))
