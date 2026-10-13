"""Read-only system probes and bounded crash evidence, without ML imports."""
import csv
import ctypes as C
import ctypes.util
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

PROFILES = {'offscreen': ['-RenderOffScreen'], 'windowed': [],
            'low': ['-quality-level=Low'], 'low_offscreen': ['-quality-level=Low', '-RenderOffScreen']}


def engine_root(config, settings=None):
    selected = (settings or config.settings()).get('carla_install_path', '')
    return Path(selected).expanduser().resolve() if selected else config.path('carla')


def command_probe(command, timeout=8):
    try:
        p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=timeout)
        return {'exit_code': p.returncode, 'stdout': p.stdout[-24000:], 'stderr': p.stderr[-3000:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'exit_code': None, 'stdout': '', 'stderr': str(error)}


def vulkan_probe():
    """Enumerate actual ICD physical devices even when vulkaninfo is absent."""
    class CreateInfo(C.Structure):
        _fields_ = [('sType', C.c_uint32), ('pNext', C.c_void_p), ('flags', C.c_uint32),
                    ('application', C.c_void_p), ('layerCount', C.c_uint32), ('layers', C.c_void_p),
                    ('extensionCount', C.c_uint32), ('extensions', C.c_void_p)]
    instance = C.c_void_p()
    try:
        lib = C.CDLL(ctypes.util.find_library('vulkan') or 'libvulkan.so.1')
        lib.vkCreateInstance.argtypes = [C.POINTER(CreateInfo), C.c_void_p, C.POINTER(C.c_void_p)]
        lib.vkEnumeratePhysicalDevices.argtypes = [C.c_void_p, C.POINTER(C.c_uint32), C.c_void_p]
        lib.vkGetPhysicalDeviceProperties.argtypes = [C.c_void_p, C.c_void_p]
        lib.vkDestroyInstance.argtypes = [C.c_void_p, C.c_void_p]
        rc = lib.vkCreateInstance(C.byref(CreateInfo(sType=1)), None, C.byref(instance))
        if rc: raise RuntimeError('vkCreateInstance returned {}'.format(rc))
        count = C.c_uint32()
        rc = lib.vkEnumeratePhysicalDevices(instance, C.byref(count), None)
        if rc or not count.value: raise RuntimeError('No Vulkan physical device: {}'.format(rc))
        devices = (C.c_void_p * count.value)()
        rc = lib.vkEnumeratePhysicalDevices(instance, C.byref(count), devices)
        if rc: raise RuntimeError('vkEnumeratePhysicalDevices returned {}'.format(rc))
        rows = []
        for device in devices:
            # VkPhysicalDeviceProperties begins with 5 uint32 fields and name[256].
            # The remaining limits/alignment fields are stored in a generous buffer.
            data = C.create_string_buffer(4096)
            lib.vkGetPhysicalDeviceProperties(device, data)
            head = (C.c_uint32 * 5).from_buffer(data)
            api = head[0]
            rows.append({'name': data.raw[20:276].split(b'\0')[0].decode(errors='replace'),
                         'vendor_id': head[2], 'device_type': head[4],
                         'api_version': '{}.{}.{}'.format(api >> 22, (api >> 12) & 1023, api & 4095),
                         'driver_raw': head[1]})
        return {'available': any(x['device_type'] != 4 for x in rows), 'devices': rows,
                'nvidia_available': any(x['vendor_id'] == 0x10de and x['device_type'] != 4 for x in rows)}
    except Exception as error:
        return {'available': False, 'devices': [], 'error': str(error)}
    finally:
        if instance.value:
            lib.vkDestroyInstance(instance, None)


def diagnostics(config, settings=None):
    s = settings or config.settings()
    root = engine_root(config, s)
    gpu = command_probe(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used', '--format=csv,noheader,nounits'])
    rows = []
    if gpu['exit_code'] == 0:
        for row in csv.reader(io.StringIO(gpu['stdout'])):
            if len(row) == 4:
                rows.append(dict(zip(('name', 'driver_version', 'memory_total_mib', 'memory_used_mib'), [x.strip() for x in row])))
    vulkan = vulkan_probe()
    binary = root / 'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'
    version = (root / 'VERSION').read_text().strip() if (root / 'VERSION').is_file() else None
    warnings = []
    if not rows: warnings.append('NVIDIA GPU 조회 실패: ' + gpu['stderr'])
    if not vulkan.get('nvidia_available'): warnings.append('NVIDIA Vulkan 장치를 열 수 없습니다. ICD/userspace 라이브러리를 확인하세요.')
    if s.get('carla_render_profile', 'offscreen') in ('low','low_offscreen'): warnings.append('CARLA 0.9.15 Low 품질은 이 장비의 지도 전환에서 segfault가 재현됐습니다. Epic 프로파일을 권장합니다.')
    return {'checked_at': time.time(), 'carla_version': version, 'unreal_version': '4.26.2 (CARLA 0.9.15)',
            'install_path': str(root), 'binary_exists': binary.is_file(), 'executable': os.access(str(binary), os.X_OK),
            'gpu': rows, 'gpu_probe': gpu, 'vulkan': vulkan,
            'gpu_processes': command_probe(['nvidia-smi'])['stdout'],
            'icd_files': [str(p) for folder in ('/usr/share/vulkan/icd.d','/etc/vulkan/icd.d') for p in Path(folder).glob('*.json')],
            'kernel': platform.release(), 'os_release': Path('/etc/os-release').read_text(),
            'session': {k: os.environ.get(k) for k in ('DISPLAY','WAYLAND_DISPLAY','XDG_SESSION_TYPE')},
            'memory': command_probe(['free','-h'])['stdout'], 'disk': command_probe(['df','-h',str(root.parent),'/dev/shm'])['stdout'],
            'modules': command_probe(['lsmod'])['stdout'].splitlines()[:1] + [x for x in command_probe(['lsmod'])['stdout'].splitlines() if 'nvidia' in x or 'nouveau' in x],
            'virtualization': command_probe(['systemd-detect-virt'])['stdout'].strip(),
            'render_profile': s.get('carla_render_profile', 'offscreen'), 'warnings': warnings,
            'ready': version == '0.9.15' and binary.is_file() and os.access(str(binary),os.X_OK) and bool(rows) and vulkan.get('nvidia_available',False)}


def crash_evidence(config, settings, run, exit_code, stage):
    run = Path(run)
    evidence = {'exit_code': exit_code, 'last_stage': stage, 'environment': diagnostics(config, settings),
                'kernel': command_probe(['journalctl','-k','--since','-15min','--no-pager']),
                'coredumps': command_probe(['coredumpctl','info','CarlaUE4-Linux-Shipping','--no-pager']), 'unreal_files': []}
    root = engine_root(config, settings)
    for folder in (root/'CarlaUE4/Saved/Logs', root/'CarlaUE4/Saved/Crashes'):
        if folder.is_dir():
            for p in sorted(folder.rglob('*'), key=lambda p:p.stat().st_mtime, reverse=True)[:20]:
                if p.is_file():
                    evidence['unreal_files'].append({'path': str(p), 'bytes': p.stat().st_size})
                    if p.stat().st_size <= 5*1024*1024:
                        target = run/'unreal_evidence'/p.relative_to(root)
                        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(str(p),str(target))
    from .config import atomic_json
    atomic_json(run/'diagnostics.json', evidence)
    return evidence
