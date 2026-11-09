"""Start a local vLLM server for the duration of a run."""
from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Iterator, Optional, Sequence

logger = logging.getLogger(__name__)

# Local LLM weights are stored under $HOME/.cache of this directory, not in ~/.cache.
DEFAULT_LLM_HOME = "/lnet/work/people/jon/llm_agents"
# These would take precedence over $HOME/.cache and put the models somewhere else.
_CACHE_OVERRIDES = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _responding(url: str) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


@contextmanager
def managed_vllm(
    model: str,
    executable: str = "vllm",
    port: Optional[int] = None,
    gpus: int = 1,
    max_model_len: Optional[int] = None,
    extra_args: Sequence[str] = (),
    log_path: str = "vllm.log",
    startup_timeout: float = 3600.0,
    home: str = DEFAULT_LLM_HOME,
) -> Iterator[str]:
    """Yield the base URL of a vLLM server that is stopped again on exit."""
    port = port or _free_port()
    command = [executable, "serve", model, "--port", str(port), "--host", "127.0.0.1",
               "--tensor-parallel-size", str(gpus)]
    if max_model_len:
        command += ["--max-model-len", str(max_model_len)]
    command += list(extra_args)
    logger.info("Starting vLLM (log: %s): %s", log_path, " ".join(command))

    base_url = f"http://127.0.0.1:{port}/v1"
    env = {k: v for k, v in os.environ.items() if k not in _CACHE_OVERRIDES}
    env["HOME"] = home
    with open(log_path, "ab") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    try:
        deadline = time.monotonic() + startup_timeout
        while not _responding(f"{base_url}/models"):
            if process.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {process.returncode} during startup, see {log_path}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"vLLM not ready after {startup_timeout:.0f} s, see {log_path}")
            time.sleep(min(5.0, max(0.2, startup_timeout / 100)))
        logger.info("vLLM ready at %s", base_url)
        yield base_url
    finally:
        if process.poll() is None:
            # vLLM spawns workers, so signal its whole process group.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
