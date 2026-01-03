from __future__ import annotations

import base64
import math
import threading
import os
import time
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
    # AgentGym-only convenience: unlock the desktop after reset (does not modify OSWorld code).
    client_password: Optional[str] = None
    auto_unlock: Optional[bool] = None
    # AgentGym-specific: automatically call OSWorld DesktopEnv.evaluate() after each step
    # and mark done when score reaches the threshold.
    auto_evaluate: Optional[bool] = None
    done_score_threshold: Optional[float] = None
    # AgentGym-specific: if true, treat "finished()/DONE" as a hard stop regardless of evaluation score.
    # Evaluation (if available) may still be computed and returned as reward/score, but will not gate termination.
    force_done_on_finished: Optional[bool] = None


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
        self._client_password: Dict[int, str] = {}
        self._auto_unlock: Dict[int, bool] = {}
        self._force_done_on_finished: Dict[int, bool] = {}
        # If the agent calls DONE but evaluation fails, remember it and reject repeated DONE
        # until a non-DONE action happens (prevents infinite finished()->finished() loops).
        self._done_rejected: Dict[int, bool] = {}
        self._last_rejected_score: Dict[int, float] = {}
        self._last_action_repr: Dict[int, str] = {}
        self._repeat_action_count: Dict[int, int] = {}

    def _maybe_unlock_screen(self, env: DesktopEnv, password: str) -> None:
        ctrl = getattr(env, "controller", None)
        if ctrl is None or not hasattr(ctrl, "execute_python_command"):
            return
        pwd = (password or "").strip()
        # Best-effort unlock:
        # - wake lock screen
        # - open password prompt
        # - try blank password, then provided password (if any)
        # - dismiss stray overlays afterwards
        cmd = "\n".join(
            [
                "import time",
                "import pyautogui",
                "pyautogui.FAILSAFE = False",
                "time.sleep(0.2)",
                "pyautogui.press('esc')",
                "time.sleep(0.2)",
                "pyautogui.press('esc')",
                "time.sleep(0.2)",
                "pyautogui.press('enter')",
                "time.sleep(0.4)",
                f"pwd = {pwd!r}",
                "if not pwd:",
                "    pyautogui.press('enter')",
                "    time.sleep(0.6)",
                "else:",
                "    pyautogui.typewrite(pwd)",
                "    time.sleep(0.1)",
                "    pyautogui.press('enter')",
                "    time.sleep(0.8)",
                "pyautogui.press('esc')",
                "time.sleep(0.1)",
                "pyautogui.press('esc')",
            ]
        )
        try:
            ctrl.execute_python_command(cmd)
        except Exception:
            return

    def _next_id(self) -> int:
        with self._lock:
            env_id = self._max_id
            self._max_id += 1
            return env_id

    def create(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = cfg or {}
        env_id = self._next_id()

        # 使用 OSWorld DesktopEnv 默认参数，允许通过 cfg 覆盖。
        # NOTE: 不要在 /create 阶段做 env.reset() 这类“启动模拟器/回滚快照”的重操作：
        # - 会让 /create 阻塞很久，客户端（AgentGym）在初始化时容易 ReadTimeout；
        # - 真正需要启动环境的是后续的 /reset(task_config=...)（BaseTask 会先 reset 再 observe）。
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
        # Some OSWorld evaluators (e.g., Chrome getters) expect these proxy flags to exist on env.
        # AgentGym's server path doesn't manage proxy, so default them to False to avoid AttributeError.
        if not hasattr(env, "current_use_proxy"):
            setattr(env, "current_use_proxy", False)
        if not hasattr(env, "enable_proxy"):
            setattr(env, "enable_proxy", False)
        self.env[env_id] = env
        self.ls.append(env_id)
        self._auto_evaluate[env_id] = bool(cfg.get("auto_evaluate")) if cfg.get("auto_evaluate") is not None else False
        self._done_score_threshold[env_id] = float(cfg.get("done_score_threshold")) if cfg.get("done_score_threshold") is not None else 1.0
        self._client_password[env_id] = str(cfg.get("client_password") or "")
        self._auto_unlock[env_id] = True if cfg.get("auto_unlock") is None else bool(cfg.get("auto_unlock"))
        # "finished()/DONE" behavior:
        # - Default: hard-stop on DONE (what AgentGym wants for offline rollouts / debug loops).
        # - Override: pass create_cfg.force_done_on_finished=false or set env var to a falsy value.
        env_default = os.environ.get("AGENTGYM_OSWORLD_FORCE_DONE_ON_FINISHED", "").strip()
        if cfg.get("force_done_on_finished") is None:
            if env_default == "":
                self._force_done_on_finished[env_id] = True
            else:
                self._force_done_on_finished[env_id] = env_default in {"1", "true", "True", "yes", "Y"}
        else:
            self._force_done_on_finished[env_id] = bool(cfg.get("force_done_on_finished"))
        self.info[env_id] = {
            "observation": {},
            "reward": 0.0,
            "done": False,
        }
        logger.info("CREATE id=%s", env_id)
        return {"id": env_id}

    def step(self, env_id: int, action: Any, pause: float = 2) -> Dict[str, Any]:
        env = self.env[env_id]
        # Detect obvious "stuck" loops (same action repeated many times).
        # We do not force-stop the episode; we only surface feedback to the agent to change strategy.
        try:
            cur_repr = repr(action)
            if self._last_action_repr.get(env_id) == cur_repr:
                self._repeat_action_count[env_id] = int(self._repeat_action_count.get(env_id, 1)) + 1
            else:
                self._repeat_action_count[env_id] = 1
            self._last_action_repr[env_id] = cur_repr
        except Exception:
            pass

        # AgentGym-side extension:
        # Allow a narrow class of raw `pyautogui...` python snippets (generated by our UI-TARS adapter)
        # even when OSWorld is running with `action_space="computer_13"`.
        # OSWorld's `DesktopEnv.step()` would otherwise route strings to `controller.execute_action()` and crash.
        if (
            isinstance(action, str)
            and env.action_space == "computer_13"
            and ("pyautogui." in action or action.lstrip().startswith("import pyautogui"))
        ):
            reward, done, info = 0.0, False, {}
            try:
                env._step_no += 1  # noqa: SLF001
                env.action_history.append(action)  # type: ignore[arg-type]
            except Exception:
                pass
            ctrl = getattr(env, "controller", None)
            if ctrl is None or not hasattr(ctrl, "execute_python_command"):
                raise RuntimeError("controller.execute_python_command not available for pyautogui action")
            ctrl.execute_python_command(action)
            time.sleep(pause)
            ob = env._get_obs()  # noqa: SLF001
        else:
            ob, reward, done, info = env.step(action, pause=pause)
        ob_json = _sanitize_observation(ob)
        score = None
        eval_error: Optional[str] = None
        eval_feedback: Optional[str] = None
        auto_eval = bool(self._auto_evaluate.get(env_id, False))
        already_scored = isinstance(info, dict) and ("score" in info)
        requested_done = (
            action == "DONE"
            or (isinstance(action, dict) and action.get("action_type") == "DONE")
        )
        if already_scored:
            try:
                score = float(info.get("score"))  # type: ignore[call-arg]
            except Exception:
                score = None

        # Clear reject state when the agent actually takes a non-DONE action.
        if not requested_done:
            self._done_rejected.pop(env_id, None)
            self._last_rejected_score.pop(env_id, None)

        # If requested: always terminate on DONE, regardless of score.
        if requested_done and bool(self._force_done_on_finished.get(env_id, False)):
            if hasattr(env, "evaluator"):
                try:
                    raw = env.evaluate()
                    score = float(raw)
                    if math.isfinite(score):
                        reward = float(score)
                    else:
                        raise ValueError(f"Non-finite evaluation score: {raw!r}")
                except Exception as e:
                    eval_error = f"{type(e).__name__}: {e}"
                    logger.exception("EVAL_ON_FORCE_DONE_FAILED id=%s", env_id)
            done = True
        # Otherwise, gate "finished()/DONE": only terminate when evaluator confirms success.
        # Gate "finished()/DONE": only terminate when evaluator confirms success.
        # This prevents the model from ending the episode prematurely and losing the chance to continue.
        elif requested_done and hasattr(env, "evaluator"):
            if bool(self._done_rejected.get(env_id, False)):
                # Repeated DONE without any other actions: don't re-run expensive evaluators.
                score = float(self._last_rejected_score.get(env_id, 0.0))
                done = False
                eval_feedback = (
                    f"Finished was rejected (score={score:.3f}); take more actions before calling finished() again."
                )
            else:
                try:
                    raw = env.evaluate()
                    score = float(raw)
                    if not math.isfinite(score):
                        raise ValueError(f"Non-finite evaluation score: {raw!r}")
                    reward = float(score)
                    threshold = float(self._done_score_threshold.get(env_id, 1.0))
                    done = bool(score >= threshold)
                    if not done:
                        self._done_rejected[env_id] = True
                        self._last_rejected_score[env_id] = float(score)
                        eval_feedback = (
                            f"Finished was rejected (score={score:.3f}); continue working and ensure the task requirements are met."
                        )
                    else:
                        self._done_rejected.pop(env_id, None)
                        self._last_rejected_score.pop(env_id, None)
                except Exception as e:
                    eval_error = f"{type(e).__name__}: {e}"
                    logger.exception("EVAL_ON_DONE_FAILED id=%s", env_id)

        # If we are looping on the same action, attach a hint so the model stops repeating it.
        try:
            rep_cnt = int(self._repeat_action_count.get(env_id, 0))
            if not done and rep_cnt >= 6 and not eval_feedback:
                eval_feedback = (
                    f"Action seems repeated {rep_cnt} times; likely no effect. "
                    f"Try a different action (e.g., click a different UI element, or use drag instead of scroll)."
                )
        except Exception:
            pass

        if auto_eval and not already_scored:
            if hasattr(env, "evaluator"):
                # If the evaluator has postconfig steps, evaluating on every step can be extremely
                # expensive (e.g., LibreOffice exports) and may even fail due to timeouts/locks.
                # In that case, only evaluate when the episode is already terminal.
                try:
                    needs_postconfig = bool(getattr(env, "evaluator", {}).get("postconfig"))
                except Exception:
                    needs_postconfig = False
                if needs_postconfig and not done:
                    # Skip mid-episode evaluation; keep reward/done as returned by env.step().
                    pass
                else:
                    # Some evaluators (e.g., Chrome getters) expect these proxy flags to exist on env.
                    # AgentGym's server path doesn't manage proxy, so default them to False.
                    if not hasattr(env, "current_use_proxy"):
                        setattr(env, "current_use_proxy", False)
                    if not hasattr(env, "enable_proxy"):
                        setattr(env, "enable_proxy", False)
                    try:
                        raw = env.evaluate()
                        score = float(raw)
                        if not math.isfinite(score):
                            raise ValueError(f"Non-finite evaluation score: {raw!r}")
                        reward = score
                        threshold = float(self._done_score_threshold.get(env_id, 1.0))
                        if score >= threshold:
                            done = True
                    except Exception as e:
                        eval_error = f"{type(e).__name__}: {e}"
                        logger.exception("AUTO_EVAL_FAILED id=%s", env_id)
                        # Make eval failure impossible to miss downstream: return a sentinel reward and stop this episode.
                        reward = -1.0
                        done = True
                        try:
                            info = dict(info) if isinstance(info, dict) else {"info": info}
                            info["eval_error"] = eval_error
                        except Exception:
                            pass
            else:
                eval_error = "auto_evaluate is enabled but env.evaluator is not set; did you /reset with task_config?"
                logger.error("AUTO_EVAL_FAILED id=%s reason=%s", env_id, eval_error)
                reward = -1.0
                done = True
                try:
                    info = dict(info) if isinstance(info, dict) else {"info": info}
                    info["eval_error"] = eval_error
                except Exception:
                    pass
        elif auto_eval and already_scored:
            # The underlying OSWorld env already evaluated on terminal actions (DONE/FAIL) and
            # attached the score in info; just propagate it and apply threshold logic.
            try:
                if score is not None and math.isfinite(float(score)):
                    reward = float(score)
                    threshold = float(self._done_score_threshold.get(env_id, 1.0))
                    if float(score) >= threshold:
                        done = True
            except Exception:
                pass
        payload = {
            "observation": ob_json,
            "reward": float(reward),
            "done": bool(done),
        }
        if auto_eval:
            payload["eval_ok"] = bool(eval_error is None)
        if score is not None:
            payload["score"] = float(score)
            payload["success"] = bool(payload["done"])
        if eval_error:
            payload["eval_error"] = eval_error
        if eval_feedback:
            payload["eval_feedback"] = eval_feedback
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
        # Ensure proxy flags exist before any evaluation.
        if not hasattr(env, "current_use_proxy"):
            setattr(env, "current_use_proxy", False)
        if not hasattr(env, "enable_proxy"):
            setattr(env, "enable_proxy", False)
        ob = env.reset(task_config=task_config, seed=seed)
        # Best-effort unlock after reset (some snapshots boot into lock screen before GNOME settings apply).
        try:
            if bool(self._auto_unlock.get(env_id, True)):
                self._maybe_unlock_screen(env, self._client_password.get(env_id, ""))
        except Exception:
            pass
        ob_json = _sanitize_observation(ob)
        payload: Dict[str, Any] = {
            "observation": ob_json,
            "reward": 0.0,
            "done": False,
        }
        auto_eval = bool(self._auto_evaluate.get(env_id, False))
        if auto_eval:
            if hasattr(env, "evaluator"):
                try:
                    needs_postconfig = bool(getattr(env, "evaluator", {}).get("postconfig"))
                    if needs_postconfig:
                        # Avoid running heavy postconfig immediately on reset.
                        payload["eval_ok"] = True
                        self.info[env_id] = payload
                        try:
                            obs_keys = list(ob_json.keys()) if isinstance(ob_json, dict) else []
                            logger.info("RESET id=%s obs_keys=%s", env_id, obs_keys)
                        except Exception:
                            pass
                        return payload
                    raw = env.evaluate()
                    score = float(raw)
                    if not math.isfinite(score):
                        raise ValueError(f"Non-finite evaluation score: {raw!r}")
                    payload["reward"] = float(score)
                    threshold = float(self._done_score_threshold.get(env_id, 1.0))
                    payload["done"] = bool(score >= threshold)
                    payload["score"] = float(score)
                    payload["success"] = bool(payload["done"])
                    payload["eval_ok"] = True
                except Exception as e:
                    eval_error = f"{type(e).__name__}: {e}"
                    logger.exception("AUTO_EVAL_FAILED_ON_RESET id=%s", env_id)
                    payload["reward"] = -1.0
                    payload["eval_ok"] = False
                    payload["eval_error"] = eval_error
            else:
                eval_error = "auto_evaluate is enabled but env.evaluator is not set; did you /reset with task_config?"
                logger.error("AUTO_EVAL_FAILED_ON_RESET id=%s reason=%s", env_id, eval_error)
                payload["reward"] = -1.0
                payload["eval_ok"] = False
                payload["eval_error"] = eval_error
        self.info[env_id] = payload
        try:
            obs_keys = list(ob_json.keys()) if isinstance(ob_json, dict) else []
            logger.info("RESET id=%s obs_keys=%s", env_id, obs_keys)
        except Exception:
            pass
        return payload

    def observation(self, env_id: int) -> Dict[str, Any]:
        # 实时从 Env 获取；如果还没 reset 启动 emulator/controller，则返回空 observation。
        env = self.env[env_id]
        if not hasattr(env, "controller"):
            return {"observation": {}}
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
            self._client_password.pop(env_id, None)
            self._auto_unlock.pop(env_id, None)
            self._force_done_on_finished.pop(env_id, None)
            self._done_rejected.pop(env_id, None)
            self._last_rejected_score.pop(env_id, None)
            self._last_action_repr.pop(env_id, None)
            self._repeat_action_count.pop(env_id, None)
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
