from __future__ import annotations

import base64
import threading
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys
sys.path.append('/data/pengchen/OSWorld/Osworld/OSWorld')
# 直接复用 OSWorld 的环境实现
from desktop_env.desktop_env import DesktopEnv
import logging


def _b64(data: Optional[bytes]) -> Optional[str]:
    if data is None:
        return None
    try:
        return "data:image/png;base64," + base64.b64encode(data).decode("utf-8")
    except Exception:
        return None


def _sanitize_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obs, dict):
        return {"observation": obs}
    out: Dict[str, Any] = dict(obs)
    # DesktopEnv._get_obs() 返回 bytes 截图，需要可序列化
    if "screenshot" in out:
        out["screenshot_base64"] = _b64(out.get("screenshot"))
        out.pop("screenshot", None)
    return out


class CreateRequestBody(BaseModel):
    provider_name: Optional[str] = None
    region: Optional[str] = None
    path_to_vm: Optional[str] = None
    snapshot_name: Optional[str] = None
    action_space: Optional[str] = None  # "computer_13" | "pyautogui"
    cache_dir: Optional[str] = None
    headless: Optional[bool] = None
    require_a11y_tree: Optional[bool] = None
    require_terminal: Optional[bool] = None
    os_type: Optional[str] = None
    # AgentGym-specific: automatically call OSWorld DesktopEnv.evaluate() after each step
    # and mark done when score reaches the threshold.
    auto_evaluate: Optional[bool] = None
    done_score_threshold: Optional[float] = None


class StepRequestBody(BaseModel):
    id: int
    action: Any
    pause: Optional[float] = 2


class ResetRequestBody(BaseModel):
    id: int
    task_config: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None


class CloseRequestBody(BaseModel):
    id: int


class RecordingRequestBody(BaseModel):
    id: int
    dest: Optional[str] = None


class OSWorldServer:
    """环境服务器包装器，管理多个 OSWorld DesktopEnv 实例。"""

    def __init__(self):
        self._max_id: int = 0
        self._lock = threading.Lock()
        self.env: Dict[int, DesktopEnv] = {}
        self.info: Dict[int, Dict[str, Any]] = {}
        self.ls: list[int] = []
        self._auto_evaluate: Dict[int, bool] = {}
        self._done_score_threshold: Dict[int, float] = {}

    def _next_id(self) -> int:
        with self._lock:
            env_id = self._max_id
            self._max_id += 1
            return env_id

    def create(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = cfg or {}
        env_id = self._next_id()

        # 使用 OSWorld DesktopEnv 默认参数，允许通过 cfg 覆盖
        env = DesktopEnv(
            provider_name=(cfg.get("provider_name") or "docker"),
            region=cfg.get("region"),
            path_to_vm=cfg.get("path_to_vm"),
            snapshot_name=(cfg.get("snapshot_name") or "init_state"),
            action_space=(cfg.get("action_space") or "pyautogui"),
            cache_dir=(cfg.get("cache_dir") or "cache"),
            screen_size=(cfg.get("screen_size") or (1920, 1080)),
            headless=bool(cfg.get("headless")) if cfg.get("headless") is not None else False,
            require_a11y_tree=True if cfg.get("require_a11y_tree") is None else bool(cfg.get("require_a11y_tree")),
            require_terminal=False if cfg.get("require_terminal") is None else bool(cfg.get("require_terminal")),
            os_type=(cfg.get("os_type") or "Ubuntu"),
        )
        # 调用 reset 以在 docker/provider 下启动模拟器（controller 等在此初始化）
        ob = env.reset(task_config=None)
        self.env[env_id] = env
        self.ls.append(env_id)
        self._auto_evaluate[env_id] = bool(cfg.get("auto_evaluate")) if cfg.get("auto_evaluate") is not None else False
        self._done_score_threshold[env_id] = float(cfg.get("done_score_threshold")) if cfg.get("done_score_threshold") is not None else 1.0
        self.info[env_id] = {
            "observation": _sanitize_observation(ob),
            "reward": 0.0,
            "done": False,
        }
        logger.info("CREATE id=%s", env_id)
        return {"id": env_id}

    def step(self, env_id: int, action: Any, pause: float = 2) -> Dict[str, Any]:
        env = self.env[env_id]
        ob, reward, done, info = env.step(action, pause=pause)
        ob_json = _sanitize_observation(ob)
        score = None
        eval_error: Optional[str] = None
        if self._auto_evaluate.get(env_id, False) and hasattr(env, "evaluator"):
            try:
                raw = env.evaluate()
                score = float(env.evaluate())
                reward = score
                threshold = float(self._done_score_threshold.get(env_id, 1.0))
                if score >= threshold:
                    done = True
            except Exception as e:
                eval_error = str(e)
                try:
                    info = dict(info) if isinstance(info, dict) else {"info": info}
                    info["eval_error"] = str(e)
                except Exception:
                    pass
        elif self._auto_evaluate.get(env_id, False) and not hasattr(env, "evaluator"):
            eval_error = "auto_evaluate is enabled but env.evaluator is not set; did you /reset with task_config?"
        payload = {
            "observation": ob_json,
            "reward": float(reward),
            "done": bool(done),
        }
        if score is not None:
            payload["score"] = float(score)
            payload["success"] = bool(payload["done"])
        if eval_error:
            payload["eval_error"] = eval_error
        # 缓存最近一次摘要
        try:
            self.info[env_id] = payload | {"info": info, "action": action}
        except Exception:
            self.info[env_id] = payload | {"info": info}
        try:
            obs_keys = list(ob_json.keys()) if isinstance(ob_json, dict) else []
            logger.info(
                "RESULT id=%s reward=%.3f done=%s obs_keys=%s",
                env_id, float(reward), bool(done), obs_keys,
            )
        except Exception:
            pass
        return payload

    def reset(self, env_id: int, task_config: Optional[Dict[str, Any]], seed: Optional[int]) -> Dict[str, Any]:
        env = self.env[env_id]
        ob = env.reset(task_config=task_config, seed=seed)
        ob_json = _sanitize_observation(ob)
        payload: Dict[str, Any] = {
            "observation": ob_json,
            "reward": 0.0,
            "done": False,
        }
        if self._auto_evaluate.get(env_id, False) and hasattr(env, "evaluator"):
            try:
                score = float(env.evaluate())
                payload["reward"] = float(score)
                threshold = float(self._done_score_threshold.get(env_id, 1.0))
                payload["done"] = bool(score >= threshold)
                payload["score"] = float(score)
                payload["success"] = bool(payload["done"])
            except Exception:
                pass
        self.info[env_id] = payload
        try:
            obs_keys = list(ob_json.keys()) if isinstance(ob_json, dict) else []
            logger.info("RESET id=%s obs_keys=%s", env_id, obs_keys)
        except Exception:
            pass
        return payload

    def observation(self, env_id: int) -> Dict[str, Any]:
        # 实时从 Env 获取
        env = self.env[env_id]
        ob = env._get_obs()  # noqa: SLF001 保持与 OSWorld 一致
        return {"observation": _sanitize_observation(ob)}

    def last(self, env_id: int) -> Dict[str, Any]:
        return self.info.get(env_id, {})

    def close(self, env_id: int) -> Dict[str, Any]:
        try:
            if env_id in self.ls:
                self.ls.remove(env_id)
            env = self.env.pop(env_id)
            env.close()
            self.info.pop(env_id, None)
            self._auto_evaluate.pop(env_id, None)
            self._done_score_threshold.pop(env_id, None)
            return {"closed": True}
        except KeyError:
            return {"closed": False, "error": "Env not exist"}
        except Exception as e:  # pylint: disable=broad-except
            return {"closed": False, "error": str(e)}

    def start_recording(self, env_id: int) -> Dict[str, Any]:
        env = self.env[env_id]
        ctrl = getattr(env, "controller", None)
        if ctrl is None or not hasattr(ctrl, "start_recording"):
            return {"ok": False, "error": "controller.start_recording not available"}
        try:
            ctrl.start_recording()
            return {"ok": True}
        except Exception as e:  # pylint: disable=broad-except
            return {"ok": False, "error": str(e)}

    def end_recording(self, env_id: int, dest: str) -> Dict[str, Any]:
        env = self.env[env_id]
        ctrl = getattr(env, "controller", None)
        if ctrl is None or not hasattr(ctrl, "end_recording"):
            return {"ok": False, "error": "controller.end_recording not available"}
        if not dest:
            return {"ok": False, "error": "dest is required"}
        try:
            ctrl.end_recording(dest)
            return {"ok": True, "dest": dest}
        except Exception as e:  # pylint: disable=broad-except
            return {"ok": False, "error": str(e)}


app = FastAPI()
server = OSWorldServer()
logger = logging.getLogger("agent_osworld.server")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@app.get("/")
def hello():
    return "This is environment OSWorld."


@app.post("/create")
def create(body: Optional[CreateRequestBody] = None):
    try:
        cfg = body.model_dump() if body else {}
        return server.create(cfg)
    except Exception as e:  # 返回详细错误，便于定位 500
        import traceback
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        })


@app.post("/step")
def step(body: StepRequestBody):
    try:
        from pprint import pformat
        logger.info("STEP id=%s action=%s pause=%s", body.id, pformat(body.action), body.pause)
    except Exception:
        pass
    return server.step(body.id, body.action, pause=body.pause or 2)


@app.post("/reset")
def reset(body: ResetRequestBody):
    return server.reset(body.id, body.task_config, body.seed)


@app.get("/observation")
def get_observation(id: int):  # noqa: A002 - 保持 API 语义
    return server.observation(id)


@app.post("/close")
def close(body: CloseRequestBody):
    return server.close(body.id)


@app.post("/start_recording")
def start_recording(body: RecordingRequestBody):
    return server.start_recording(body.id)


@app.post("/end_recording")
def end_recording(body: RecordingRequestBody):
    return server.end_recording(body.id, body.dest or "")

@app.get("/last")
def get_last(id: int):  # noqa: A002 - 保持 API 语义
    return server.last(id)


@app.get("/health")
def health():
    return {"status": "ok"}
