"""Small current-limited reflection of YAM motor effort onto XL330 leaders.

Motor effort includes gravity/friction; this is not a calibrated contact-force sensor.
Control table: https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/
"""
from dataclasses import dataclass
import numpy as np

CAP_MA=25

@dataclass
class HapticTarget:
    acquired_at: float
    current_ma: np.ndarray
    enabled: bool = True


def reflected_current(arm,sample,now,strength=.3):
    if sample is None or not 0<=now-sample.acquired_at<=.1 or sample.effort is None:
        return None
    effort=np.asarray(sample.effort,dtype=float).copy()
    if effort.shape!=(7,) or not np.isfinite(effort).all():return None
    grip_direction=sample.raw.get('gripper_direction')
    if grip_direction not in (-1,1):return None
    effort[-1]*=grip_direction
    return np.clip(-np.asarray(arm.signs)*effort*5.*np.clip(strength,0,1),-CAP_MA,CAP_MA)


class HapticOutput:
    """Only the GELLO acquisition thread owns this connection and its writes."""
    def __init__(self,connection,ids):
        self.connection=connection;self.ids=ids;self.original={};self.active=False
        self.current=np.zeros(len(ids),dtype=int)

    def _read(self,motor,address,size):
        packet=self.connection.packetHandler
        method=packet.read1ByteTxRx if size==1 else packet.read2ByteTxRx
        value,result,error=method(self.connection.portHandler,motor,address)
        if result or error:raise ConnectionError(f'GELLO {motor} read {address}: {result}/{error}')
        return value

    def _write(self,motor,address,value,size=1):
        packet=self.connection.packetHandler
        method=packet.write1ByteTxRx if size==1 else packet.write2ByteTxRx
        result,error=method(self.connection.portHandler,motor,address,int(value)&((1<<(8*size))-1))
        if result or error:raise ConnectionError(f'GELLO {motor} write {address}: {result}/{error}')

    def enable(self):
        if self.active:return
        try:
            for motor in self.ids:
                model,result,error=self.connection.packetHandler.ping(self.connection.portHandler,motor)
                if result or error or model not in (1190,1200):
                    raise RuntimeError(f'GELLO {motor}: unsupported or unavailable XL330 ({model})')
                self.original[motor]=(self._read(motor,11,1),self._read(motor,38,2))
                self._write(motor,64,0)
                self._write(motor,98,0)
                self._write(motor,11,0)
                self._write(motor,38,min(CAP_MA,self.original[motor][1]),2)
                self._write(motor,102,0,2)
                self._write(motor,98,5)  # Hardware watchdog: 100 ms without bus instructions.
                self._write(motor,64,1)
                if self._read(motor,64,1)!=1 or self._read(motor,11,1)!=0 or self._read(motor,38,2)>CAP_MA:
                    raise RuntimeError(f'GELLO {motor}: haptic mode verification failed')
            self.active=True
        except Exception:
            self.close()
            raise

    def write(self,current):
        if not self.active:raise RuntimeError('Haptic output is disabled')
        current=np.asarray(current)
        if current.shape!=(len(self.ids),) or not np.isfinite(current).all():
            raise ValueError('Invalid haptic current')
        for index,motor in enumerate(self.ids):
            limit=min(CAP_MA,self.original[motor][1])
            value=int(np.clip(current[index],-limit,limit))
            self._write(motor,102,value,2)
            self.current[index]=value

    def close(self):
        errors=[]
        self.active=False
        for motor,(mode,limit) in list(self.original.items()):
            try:
                # Disable torque first, even if the watchdog has made goals read-only.
                self._write(motor,64,0)
                if self._read(motor,64,1)!=0:raise RuntimeError('torque-off not confirmed')
                self._write(motor,98,0)
                self._write(motor,102,0,2)
                self._write(motor,11,mode)
                self._write(motor,38,limit,2)
                del self.original[motor]
            except Exception as error:errors.append(f'{motor}: {error}')
        self.current[:]=0
        if errors:raise RuntimeError('GELLO release unconfirmed: '+'; '.join(errors))
