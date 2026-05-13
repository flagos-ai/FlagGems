"""GPU device manager with lock-file based allocation."""

import logging
import os
import subprocess
import time

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages GPU allocation using lock files to prevent conflicts."""

    def __init__(self, lock_dir: str, gpu_ids: list[int] | None = None):
        self.lock_dir = lock_dir
        os.makedirs(lock_dir, exist_ok=True)

        if gpu_ids is not None:
            self.gpu_ids = gpu_ids
        else:
            self.gpu_ids = self._detect_gpus()

        logger.info(f"DeviceManager initialized with GPUs: {self.gpu_ids}")

    def _detect_gpus(self) -> list[int]:
        """Detect available GPUs via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                ids = [int(line.strip()) for line in result.stdout.strip().split("\n") if line.strip()]
                return ids
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        logger.warning("Failed to detect GPUs, defaulting to [0]")
        return [0]

    def _lock_path(self, gpu_id: int) -> str:
        return os.path.join(self.lock_dir, f"gpu_{gpu_id}.lock")

    def acquire(self) -> int | None:
        """Acquire a free GPU. Returns gpu_id or None if all busy."""
        for gpu_id in self.gpu_ids:
            lock_path = self._lock_path(gpu_id)
            if os.path.exists(lock_path):
                # Check if the holding process is still alive
                if self._is_lock_stale(lock_path):
                    logger.info(f"Removing stale lock for GPU {gpu_id}")
                    os.remove(lock_path)
                else:
                    continue

            # Acquire the lock
            try:
                with open(lock_path, "w") as f:
                    f.write(f"{os.getpid()}\n{time.time()}\n")
                logger.info(f"Acquired GPU {gpu_id}")
                return gpu_id
            except OSError as e:
                logger.warning(f"Failed to acquire GPU {gpu_id}: {e}")
                continue

        return None

    def release(self, gpu_id: int):
        """Release a GPU lock."""
        lock_path = self._lock_path(gpu_id)
        if os.path.exists(lock_path):
            os.remove(lock_path)
            logger.info(f"Released GPU {gpu_id}")

    def _is_lock_stale(self, lock_path: str) -> bool:
        """Check if a lock file's owning process is dead."""
        try:
            with open(lock_path) as f:
                lines = f.read().strip().split("\n")
                pid = int(lines[0])
            # Check if process is alive
            os.kill(pid, 0)
            return False
        except (OSError, ValueError, IndexError):
            return True

    def release_all(self):
        """Release all locks owned by this process."""
        for gpu_id in self.gpu_ids:
            lock_path = self._lock_path(gpu_id)
            if os.path.exists(lock_path):
                try:
                    with open(lock_path) as f:
                        pid = int(f.read().strip().split("\n")[0])
                    if pid == os.getpid():
                        os.remove(lock_path)
                        logger.info(f"Released GPU {gpu_id}")
                except (OSError, ValueError, IndexError):
                    pass

    def available_count(self) -> int:
        """Return the number of currently available GPUs."""
        count = 0
        for gpu_id in self.gpu_ids:
            lock_path = self._lock_path(gpu_id)
            if not os.path.exists(lock_path) or self._is_lock_stale(lock_path):
                count += 1
        return count
