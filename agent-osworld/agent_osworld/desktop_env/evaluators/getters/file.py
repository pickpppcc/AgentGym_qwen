import os
import logging
import time
from typing import Dict, List, Set
from typing import Optional, Any, Union
from datetime import datetime
import requests
import pandas as pd

logger = logging.getLogger("desktopenv.getter.file")


def get_content_from_vm_file(env, config: Dict[str, Any]) -> Any:
    """
    Config:
        path (str): absolute path on the VM to fetch
    """

    path = config["path"]
    file_path = get_vm_file(env, {"path": path, "dest": os.path.basename(path)})
    file_type, file_content = config['file_type'], config['file_content']
    if file_type == 'xlsx':
        if file_content == 'last_row':
            df = pd.read_excel(file_path)
            last_row = df.iloc[-1]
            last_row_as_list = last_row.astype(str).tolist()
            return last_row_as_list
    else:
        raise NotImplementedError(f"File type {file_type} not supported")


def get_cloud_file(env, config: Dict[str, Any]) -> Union[str, List[str]]:
    """
    Config:
        path (str|List[str]): the url to download from
        dest (str|List[str])): file name of the downloaded file
        multi (bool) : optional. if path and dest are lists providing
          information of multiple files. defaults to False
        gives (List[int]): optional. defaults to [0]. which files are directly
          returned to the metric. if len==1, str is returned; else, list is
          returned.
    """

    if not config.get("multi", False):
        paths: List[str] = [config["path"]]
        dests: List[str] = [config["dest"]]
    else:
        paths: List[str] = config["path"]
        dests: List[str] = config["dest"]
    cache_paths: List[str] = []

    gives: Set[int] = set(config.get("gives", [0]))
    timeout_s: float = float(config.get("timeout", 60.0))
    retries: int = int(config.get("retries", 3))
    retry_sleep_s: float = float(config.get("retry_sleep", 1.0))
    hf_mirror: str = str(
        config.get("hf_mirror")
        or os.getenv("OSWORLD_HF_MIRROR")
        or os.getenv("HF_MIRROR")
        or ""
    ).strip().rstrip("/")

    def _rewrite_hf_url(url: str) -> str:
        u = (url or "").strip()
        if not u:
            return u
        if not hf_mirror:
            return u
        if u.startswith("https://huggingface.co/") or u.startswith("http://huggingface.co/"):
            return hf_mirror + "/" + u.split("huggingface.co/", 1)[1]
        return u

    def _download(url: str, out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp_path = out_path + ".part"
        last_err: Optional[Exception] = None

        for attempt in range(max(1, retries)):
            try:
                rewritten = _rewrite_hf_url(url)
                with requests.get(rewritten, stream=True, timeout=timeout_s) as response:
                    response.raise_for_status()
                    with open(tmp_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    os.replace(tmp_path, out_path)
                    return
                raise IOError(f"Downloaded empty file for {url} -> {out_path}")
            except Exception as e:
                last_err = e
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                if attempt < max(1, retries) - 1:
                    time.sleep(retry_sleep_s)
        raise requests.RequestException(f"Failed to download after {retries} attempts: {url}") from last_err

    for i, (p, d) in enumerate(zip(paths, dests)):
        _path = os.path.join(env.cache_dir, d)
        if i in gives:
            cache_paths.append(_path)

        if os.path.exists(_path) and os.path.getsize(_path) > 0:
            #return _path
            continue

        url = p
        _download(url, _path)

    return cache_paths[0] if len(cache_paths)==1 else cache_paths


def get_vm_file(env, config: Dict[str, Any]) -> Union[Optional[str], List[Optional[str]]]:
    """
    Config:
        path (str): absolute path on the VM to fetch
        dest (str): file name of the downloaded file
        multi (bool) : optional. if path and dest are lists providing
          information of multiple files. defaults to False
        gives (List[int]): optional. defaults to [0]. which files are directly
          returned to the metric. if len==1, str is returned; else, list is
          returned.
        only support for single file now:
        time_suffix(bool): optional. defaults to False. if True, append the current time in required format.
        time_format(str): optional. defaults to "%Y%m%d_%H%M%S". format of the time suffix.
    """
    time_format = "%Y%m%d_%H%M%S"
    if not config.get("multi", False):
        paths: List[str] = [config["path"]]
        dests: List[str] = [config["dest"]]
        if config.get("time_suffix", False):
            time_format = config.get("time_format", time_format)
            # Insert time before file extension.
            dests = [f"{os.path.splitext(d)[0]}_{datetime.now().strftime(time_format)}{os.path.splitext(d)[1]}" for d in dests]
    else:
        paths: List[str] = config["path"]
        dests: List[str] = config["dest"]


    cache_paths: List[str] = []

    gives: Set[int] = set(config.get("gives", [0]))

    for i, (p, d) in enumerate(zip(paths, dests)):
        _path = os.path.join(env.cache_dir, d)
        
        try:
            # Try to get file from VM
            file = env.controller.get_file(p)
            if file is None:
                logger.warning(f"Failed to get file from VM: {p}")
                if i in gives:
                    cache_paths.append(None)
                continue

            if i in gives:
                cache_paths.append(_path)
                
            # Write file with robust error handling
            try:
                # Ensure cache directory exists
                os.makedirs(env.cache_dir, exist_ok=True)
                
                with open(_path, "wb") as f:
                    f.write(file)
                logger.info(f"Successfully saved file: {_path} ({len(file)} bytes)")
                
            except IOError as e:
                logger.error(f"IO error writing file {_path}: {e}")
                if i in gives:
                    cache_paths[-1] = None  # Replace the path we just added with None
            except Exception as e:
                logger.error(f"Unexpected error writing file {_path}: {e}")
                if i in gives:
                    cache_paths[-1] = None
                    
        except Exception as e:
            logger.error(f"Error processing file {p}: {e}")
            if i in gives:
                cache_paths.append(None)
                
    return cache_paths[0] if len(cache_paths)==1 else cache_paths


def get_cache_file(env, config: Dict[str, str]) -> str:
    """
    Config:
        path (str): relative path in cache dir
    """

    _path = os.path.join(env.cache_dir, config["path"])
    assert os.path.exists(_path)
    return _path
