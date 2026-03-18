"""SocialVAE adapter for loading an external repository and running inference."""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _load_module_from_file(module_name: str, file_path: Path, extra_path: Optional[Path] = None):
    if extra_path is not None:
        extra_path_str = str(extra_path)
        if extra_path_str not in sys.path:
            sys.path.insert(0, extra_path_str)
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_ckpt_path(ckpt_path: Path) -> Path:
    if ckpt_path.is_file():
        return ckpt_path
    if ckpt_path.is_dir():
        for name in ("ckpt-best", "ckpt-last"):
            candidate = ckpt_path / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"SocialVAE checkpoint not found: {ckpt_path}")


def _extract_state_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        for key in ("model", "model_state", "state_dict"):
            val = obj.get(key)
            if isinstance(val, dict) and val:
                return val
        if obj:
            first_val = next(iter(obj.values()))
            if TORCH_AVAILABLE and torch.is_tensor(first_val):
                return obj
    raise KeyError("Unable to extract model state_dict from checkpoint")


def _read_social_vae_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path).expanduser()
    if not path.is_file():
        return {}
    cfg_mod = _load_module_from_file("social_vae_runtime_config", path, extra_path=path.parent)
    out: Dict[str, Any] = {}
    for key in ("OB_HORIZON", "PRED_HORIZON", "OB_RADIUS", "RNN_HIDDEN_DIM"):
        if hasattr(cfg_mod, key):
            out[key] = getattr(cfg_mod, key)
    return out


def load_external_social_vae(
    repo_path: str,
    ckpt_path: str,
    device: str = "",
    config_path: str = "",
    ob_horizon: int = 8,
    pred_horizon: int = 12,
    ob_radius: float = 2.0,
    hidden_dim: int = 256,
) -> Tuple[Any, Dict[str, Any]]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for SocialVAE backend. Install torch.")

    repo = Path(repo_path).expanduser()
    if not repo.is_dir():
        raise FileNotFoundError(f"SocialVAE repo path does not exist: {repo}")

    module_file = repo / "social_vae.py"
    if not module_file.is_file():
        raise FileNotFoundError(f"Expected file not found: {module_file}")

    cfg = _read_social_vae_config(config_path)
    resolved_ob_horizon = int(cfg.get("OB_HORIZON", ob_horizon))
    resolved_pred_horizon = int(cfg.get("PRED_HORIZON", pred_horizon))
    resolved_ob_radius = float(cfg.get("OB_RADIUS", ob_radius))
    resolved_hidden_dim = int(cfg.get("RNN_HIDDEN_DIM", hidden_dim))

    module = _load_module_from_file("social_vae_runtime_module", module_file, extra_path=repo)
    social_vae_cls = getattr(module, "SocialVAE", None)
    if social_vae_cls is None:
        raise AttributeError(f"Class 'SocialVAE' not found in {module_file}")

    ctor_sig = inspect.signature(social_vae_cls)
    kwargs: Dict[str, Any] = {}
    if "horizon" in ctor_sig.parameters:
        kwargs["horizon"] = resolved_pred_horizon
    if "ob_radius" in ctor_sig.parameters:
        kwargs["ob_radius"] = resolved_ob_radius
    if "hidden_dim" in ctor_sig.parameters:
        kwargs["hidden_dim"] = resolved_hidden_dim

    try:
        model = social_vae_cls(**kwargs)
    except TypeError:
        model = social_vae_cls()

    resolved_ckpt = _resolve_ckpt_path(Path(ckpt_path).expanduser())
    ckpt = torch.load(str(resolved_ckpt), map_location="cpu")
    state_dict = _extract_state_dict(ckpt)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)

    target_device = device.strip() if device else ""
    if not target_device:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(target_device)
    model.eval()

    meta = {
        "repo_path": str(repo),
        "ckpt_path": str(resolved_ckpt),
        "device": target_device,
        "ob_horizon": resolved_ob_horizon,
        "pred_horizon": resolved_pred_horizon,
        "ob_radius": resolved_ob_radius,
        "hidden_dim": resolved_hidden_dim,
    }
    return model, meta


def _to_numpy(output: Any) -> np.ndarray:
    if isinstance(output, tuple):
        output = output[0]
    if TORCH_AVAILABLE and torch.is_tensor(output):
        return output.detach().cpu().numpy()
    if isinstance(output, np.ndarray):
        return output
    return np.asarray(output)


def predict_social_vae_samples(
    model: Any,
    x: np.ndarray,
    neighbor: np.ndarray,
    device: str,
    n_predictions: int = 20,
    expected_horizon: int = 12,
) -> np.ndarray:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for SocialVAE backend. Install torch.")
    if x.ndim != 3 or x.shape[2] != 6:
        raise ValueError(f"x must be [L,N,6], got {x.shape}")
    if neighbor.ndim != 4 or neighbor.shape[0] != x.shape[0] or neighbor.shape[1] != x.shape[1] or neighbor.shape[3] != 6:
        raise ValueError(f"neighbor must be [L,N,Nn,6], got {neighbor.shape}")

    x_t = torch.from_numpy(x.astype(np.float32)).to(device)
    neigh_t = torch.from_numpy(neighbor.astype(np.float32)).to(device)

    forward_sig = inspect.signature(model.forward)
    with torch.no_grad():
        if "n_predictions" in forward_sig.parameters:
            pred = model(x_t, neigh_t, n_predictions=max(0, int(n_predictions)))
        else:
            pred = model(x_t, neigh_t)

    arr = _to_numpy(pred)
    if arr.ndim == 3:
        if arr.shape[1] == x.shape[1] and arr.shape[2] == 2:
            arr = arr[np.newaxis, ...]
        else:
            raise ValueError(f"Unexpected SocialVAE output shape: {arr.shape}")
    if arr.ndim != 4 or arr.shape[-1] != 2:
        raise ValueError(f"Unexpected SocialVAE output shape: {arr.shape}")

    n_agents = x.shape[1]
    if arr.shape[2] != n_agents:
        if arr.shape[1] == n_agents:
            arr = np.transpose(arr, (0, 2, 1, 3))
        elif arr.shape[0] == n_agents:
            arr = np.transpose(arr, (1, 2, 0, 3))
        else:
            raise ValueError(f"Cannot align SocialVAE output to N={n_agents}: {arr.shape}")

    if arr.shape[1] == 0:
        raise ValueError("SocialVAE output has zero prediction horizon")
    if expected_horizon > 0 and arr.shape[1] != expected_horizon and arr.shape[0] == expected_horizon:
        arr = np.transpose(arr, (1, 0, 2, 3))
    return arr.astype(np.float64)
