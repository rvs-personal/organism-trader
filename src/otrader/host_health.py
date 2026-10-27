"""Small, read-only host checks for local inference; never change OS power settings."""
import os
import re
import subprocess
import sys
import time


def _read(args):
    try:
        return subprocess.check_output(args, text=True, timeout=3, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return ''


def snapshot():
    result = dict(time=time.time(), load_1m=os.getloadavg()[0], cpu_count=os.cpu_count())
    if sys.platform != 'darwin':
        return result
    pressure = _read(['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level']).strip()
    result['memory_pressure'] = {'1':'normal','2':'warning','4':'critical'}.get(pressure, 'unknown')
    swap = re.search(r'used = ([\d.]+)M', _read(['/usr/sbin/sysctl', 'vm.swapusage']))
    result['swap_used_mb'] = float(swap[1]) if swap else None
    thermal = _read(['/usr/bin/pmset', '-g', 'therm'])
    speed = re.search(r'CPU_Speed_Limit\s*=\s*(\d+)', thermal)
    result['cpu_speed_limit_pct'] = int(speed[1]) if speed else None
    result['thermal_warning'] = bool(re.search(r'(?:Thermal_Level|ThermalWarningLevel)\s*=\s*[1-9]', thermal))
    return result


def should_pause(health):
    return health.get('memory_pressure') == 'critical' or health.get('thermal_warning', False) or (
        health.get('cpu_speed_limit_pct') is not None and health['cpu_speed_limit_pct'] < 80)


def should_reduce(health):
    return should_pause(health) or health.get('memory_pressure') == 'warning'
