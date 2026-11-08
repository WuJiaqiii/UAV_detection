"""Wait for NVIDIA memory without importing torch or creating a CUDA context."""
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
import fcntl
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time


@dataclass(frozen=True)
class GpuWaitPolicy:
    # Physical nvidia-smi index/UUID, not a remapped torch device ordinal.
    gpu_id: str = '0'
    min_free_mib: int = 8192
    poll_seconds: float = 30.
    timeout_seconds: float = 0.
    oom_retries: int = 1

    def __post_init__(self):
        if not str(self.gpu_id).strip() or ',' in str(self.gpu_id):
            raise ValueError('gpu_id must select exactly one physical GPU index or UUID')
        if not isinstance(self.min_free_mib, int) or self.min_free_mib <= 0:
            raise ValueError('gpu_min_free_mib must be a positive integer')
        if not math.isfinite(self.poll_seconds) or self.poll_seconds <= 0:
            raise ValueError('gpu_poll_seconds must be finite and positive')
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds < 0:
            raise ValueError('gpu_wait_timeout must be finite and nonnegative')
        if not isinstance(self.oom_retries, int) or self.oom_retries < 0:
            raise ValueError('gpu_oom_retries must be a nonnegative integer')


@dataclass(frozen=True)
class GpuMemory:
    index: int
    uuid: str
    total_mib: int
    free_mib: int

    def as_dict(self):
        return asdict(self)


def query_gpu(gpu_id):
    command = ['nvidia-smi', '--id='+str(gpu_id),
               '--query-gpu=index,uuid,memory.total,memory.free',
               '--format=csv,noheader,nounits']
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except FileNotFoundError as exc:
        raise RuntimeError('GPU waiting requires nvidia-smi; omit --wait-for-gpu for CPU runs') from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = (getattr(exc, 'stderr', '') or str(exc)).strip()
        raise RuntimeError('Cannot query the selected GPU: '+detail) from exc
    try:
        rows = list(csv.reader(result.stdout.strip().splitlines(), skipinitialspace=True))
        if len(rows) != 1 or len(rows[0]) != 4:
            raise ValueError('Expected exactly one GPU')
        index, uuid, total, free = [x.strip() for x in rows[0]]
        gpu = GpuMemory(int(index), uuid, int(total), int(free))
        if (not gpu.uuid.startswith('GPU-') or gpu.index < 0 or
                gpu.total_mib <= 0 or not 0 <= gpu.free_mib <= gpu.total_mib):
            raise ValueError('Invalid GPU fields')
        return gpu
    except (TypeError, ValueError) as exc:
        raise RuntimeError('Unsupported nvidia-smi memory response: '+repr(result.stdout[:300])) from exc


def worker_environment(gpu):
    # Pin the worker to the SAME physical GPU that was checked, even on hosts
    # whose CUDA and nvidia-smi index order differs. Its logical ordinal is 0.
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu.uuid
    return env


def is_cuda_allocation_failure(exc):
    """Only explicit CUDA allocation errors qualify for a fresh-worker retry."""
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return ('cuda out of memory' in message or 'cuda error: out of memory' in message or
            ('cublas_status_alloc_failed' in message and 'cublascreate' in message))


@contextmanager
def reserve_gpu(policy, lock_root=None):
    """Wait indefinitely by default; keep a cooperative lock through the worker.

    The lock serializes launchers for this Unix user. Other programs can still
    allocate memory, so this is an admission check, not a VRAM reservation.
    Query/config/driver failures are actionable errors, never an endless wait.
    """
    started = time.monotonic()
    gpu = query_gpu(policy.gpu_id)
    if policy.min_free_mib > gpu.total_mib:
        raise ValueError(f'Required {policy.min_free_mib} MiB exceeds GPU total {gpu.total_mib} MiB')
    root = Path(lock_root) if lock_root is not None else Path(tempfile.gettempdir())/f'uav-gpu-{os.getuid()}'
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (root/(gpu.uuid+'.lock')).open('a') as stream:
        while True:
            elapsed = time.monotonic()-started
            if policy.timeout_seconds and elapsed >= policy.timeout_seconds:
                raise TimeoutError(f'GPU resource wait exceeded {policy.timeout_seconds:g} seconds')
            acquired = False
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                pass
            if acquired:
                try:
                    # Recheck under the lock, using the stable UUID from now on.
                    gpu = query_gpu(gpu.uuid)
                    if gpu.free_mib >= policy.min_free_mib:
                        print(f'[GPU ready] {gpu.uuid}: {gpu.free_mib}/{gpu.total_mib} MiB free; '
                              f'required {policy.min_free_mib} MiB', flush=True)
                        yield gpu
                        return
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)
                reason = f'{gpu.free_mib}/{gpu.total_mib} MiB free; required {policy.min_free_mib} MiB'
            else:
                reason = 'another UAV launcher holds this GPU lock'
            print(f'[GPU waiting {elapsed:.0f}s] {gpu.uuid}: {reason}; '
                  f'check again in {policy.poll_seconds:g}s (Ctrl+C to cancel)', flush=True)
            delay = policy.poll_seconds
            if policy.timeout_seconds:
                delay = min(delay, max(0., policy.timeout_seconds-(time.monotonic()-started)))
            time.sleep(delay)
