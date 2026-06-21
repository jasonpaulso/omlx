# SPDX-License-Identifier: Apache-2.0
"""Stdio MCP server for driving a running oMLX inference server.

This is oMLX-as-a-managed-target: an external agent (Claude Code, Claude
Desktop, any MCP host) spawns this stdio server, which forwards to a running
oMLX instance over its REST API. It is unrelated to oMLX's own MCP *client*
(omlx/mcp/, which dials out to external MCP servers) and to the browser-side
WebMCP layer (omlx/admin/static/js/webmcp/). Same protocol name, different
direction.

Configuration (environment):
    OMLX_BASE_URL   Base URL of the running server. Default http://127.0.0.1:8000.
    OMLX_API_KEY    API key. Used as a Bearer token for /v1/* and to log in to
                    the cookie-authenticated /admin/api/* surface. Optional when
                    the server runs with auth disabled.

Auth model mirrors the server's split: /v1/* takes `Authorization: Bearer
<key>`; /admin/api/* takes a session cookie obtained from POST
/admin/api/login. A single httpx client with a cookie jar holds both.

Run: uv run --with mcp --with httpx python omlx_mcp_server.py
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("omlx")


def _base_url() -> str:
    raw = (os.environ.get("OMLX_BASE_URL") or "").strip()
    # Guard against an un-expanded ${OMLX_BASE_URL} literal or empty value.
    if not raw or raw.startswith("${"):
        return "http://127.0.0.1:8000"
    return raw.rstrip("/")


def _api_key() -> str:
    raw = (os.environ.get("OMLX_API_KEY") or "").strip()
    return "" if raw.startswith("${") else raw


_client: Optional[httpx.AsyncClient] = None
_admin_ready = False


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=_base_url(), timeout=120.0)
    return _client


def _v1_headers() -> dict[str, str]:
    key = _api_key()
    return {"Authorization": f"Bearer {key}"} if key else {}


async def _ensure_admin() -> Optional[dict[str, Any]]:
    """Log in to the admin surface if needed. Returns an error dict on failure."""
    global _admin_ready
    if _admin_ready:
        return None
    key = _api_key()
    if not key:
        # No key configured — assume auth is disabled and let the call proceed.
        _admin_ready = True
        return None
    try:
        resp = await _get_client().post("/admin/api/login", json={"api_key": key})
    except httpx.HTTPError as e:
        return {"error": "CONNECTION_FAILED", "message": str(e), "base_url": _base_url()}
    if resp.status_code >= 400:
        return {
            "error": "LOGIN_FAILED",
            "status": resp.status_code,
            "message": "Admin login rejected. Check OMLX_API_KEY.",
        }
    _admin_ready = True
    return None


def _parse(resp: httpx.Response) -> Any:
    ct = resp.headers.get("content-type", "")
    return resp.json() if "application/json" in ct else resp.text


async def _v1(method: str, path: str, **kw: Any) -> Any:
    try:
        resp = await _get_client().request(method, path, headers=_v1_headers(), **kw)
    except httpx.HTTPError as e:
        return {"error": "CONNECTION_FAILED", "message": str(e), "base_url": _base_url()}
    if resp.status_code == 401:
        return {"error": "UNAUTHORIZED", "message": "Set OMLX_API_KEY to a valid key."}
    if resp.status_code >= 400:
        return {"error": "BACKEND_ERROR", "status": resp.status_code, "body": _parse(resp)}
    return _parse(resp)


async def _resolve_model() -> Optional[str]:
    """Pick a sensible model id when a caller omits one.

    Prefers the configured default, then any loaded model, then the first known
    model. Returns None if the inventory can't be read.
    """
    models = await _admin("GET", "/admin/api/models")
    items = models.get("models") if isinstance(models, dict) else None
    if not items:
        return None
    for item in items:
        if item.get("is_default"):
            return item.get("id")
    for item in items:
        if item.get("loaded"):
            return item.get("id")
    return items[0].get("id")


async def _admin(method: str, path: str, **kw: Any) -> Any:
    global _admin_ready
    err = await _ensure_admin()
    if err:
        return err
    try:
        resp = await _get_client().request(method, path, **kw)
        if resp.status_code == 401:
            # Session may have expired — re-login once and retry.
            _admin_ready = False
            err = await _ensure_admin()
            if err:
                return err
            resp = await _get_client().request(method, path, **kw)
    except httpx.HTTPError as e:
        return {"error": "CONNECTION_FAILED", "message": str(e), "base_url": _base_url()}
    if resp.status_code == 401:
        return {"error": "UNAUTHORIZED", "message": "Admin session rejected. Check OMLX_API_KEY."}
    if resp.status_code >= 400:
        return {"error": "ADMIN_API_ERROR", "status": resp.status_code, "body": _parse(resp)}
    return _parse(resp)


# ── System / health ──────────────────────────────────────────────────────────


@mcp.tool()
async def system_info() -> Any:
    """Hardware, OS, disk-free, and oMLX version for the running server.

    Merges /admin/api/device-info and /admin/api/server-info. Use to answer
    'what hardware is this?' or to check free disk before downloading a model.
    """
    device = await _admin("GET", "/admin/api/device-info")
    server = await _admin("GET", "/admin/api/server-info")
    if isinstance(device, dict) and device.get("error"):
        return device
    out: dict[str, Any] = {}
    if isinstance(device, dict):
        out.update(device)
    if isinstance(server, dict):
        out.update(server)
    return out or {"device": device, "server": server}


@mcp.tool()
async def server_stats() -> Any:
    """Live server metrics: request counts, tokens, latencies, memory, queue depth."""
    return await _admin("GET", "/admin/api/stats")


@mcp.tool()
async def tail_logs(lines: int = 200, min_level: str = "info") -> Any:
    """Return the most recent server log lines.

    Args:
        lines: How many lines to return (1-5000).
        min_level: Minimum level to include: debug, info, warning, or error.
    """
    return await _admin(
        "GET", "/admin/api/logs", params={"lines": lines, "min_level": min_level}
    )


# ── Models: inspection ────────────────────────────────────────────────────────


@mcp.tool()
async def list_models() -> Any:
    """List models currently servable at /v1/models (loaded / API-visible)."""
    return await _v1("GET", "/v1/models")


@mcp.tool()
async def list_models_detailed() -> Any:
    """Full model inventory with state: loaded/unloaded, default, pinned, size, type.

    This is the management view (/admin/api/models) — broader than list_models.
    An empty list means nothing is downloaded; use search_models + download_model.
    """
    return await _admin("GET", "/admin/api/models")


# ── Models: management ────────────────────────────────────────────────────────


@mcp.tool()
async def load_model(model_id: str) -> Any:
    """Load a downloaded model into memory. Can take 10-60s for large models."""
    return await _admin("POST", f"/admin/api/models/{model_id}/load")


@mcp.tool()
async def unload_model(model_id: str) -> Any:
    """Unload a model from memory to free RAM (e.g. before loading a larger one)."""
    return await _admin("POST", f"/admin/api/models/{model_id}/unload")


@mcp.tool()
async def reload_models() -> Any:
    """Rescan model directories and re-read per-model settings."""
    return await _admin("POST", "/admin/api/reload")


@mcp.tool()
async def set_default_model(model_id: str) -> Any:
    """Mark a model as default — used by /v1/* requests that omit `model`."""
    return await _admin(
        "PUT", f"/admin/api/models/{model_id}/settings", json={"is_default": True}
    )


@mcp.tool()
async def set_model_pinned(model_id: str, pinned: bool = True) -> Any:
    """Pin or unpin a model. Pinned models auto-load on start and resist eviction."""
    return await _admin(
        "PUT", f"/admin/api/models/{model_id}/settings", json={"is_pinned": pinned}
    )


# ── Models: download (HuggingFace) ────────────────────────────────────────────


@mcp.tool()
async def search_models(query: str, mlx_only: bool = True, sort: str = "downloads") -> Any:
    """Search HuggingFace Hub for MLX-compatible models.

    Args:
        query: Free-text terms, e.g. 'Qwen3 4B coding'.
        mlx_only: Restrict to MLX-quantized models.
        sort: downloads, likes, trending, or recent.
    """
    params = {"q": query, "sort": sort}
    if mlx_only:
        params["mlx_only"] = "true"
    return await _admin("GET", "/admin/api/hf/search", params=params)


@mcp.tool()
async def download_model(repo_id: str, token: Optional[str] = None) -> Any:
    """Start a HuggingFace download. Returns a task_id; poll download_status.

    Args:
        repo_id: HF repo, e.g. 'mlx-community/Qwen3-4B-MLX-4bit'.
        token: Optional HF token for gated repos.
    """
    body: dict[str, Any] = {"repo_id": repo_id}
    if token:
        body["token"] = token
    return await _admin("POST", "/admin/api/hf/download", json=body)


@mcp.tool()
async def download_status() -> Any:
    """List active and recent download tasks with progress. Poll until completed/failed."""
    return await _admin("GET", "/admin/api/hf/tasks")


# ── Settings & cache ──────────────────────────────────────────────────────────


@mcp.tool()
async def get_settings() -> Any:
    """Return the full global settings tree (server, memory, scheduler, sampling, etc.)."""
    return await _admin("GET", "/admin/api/global-settings")


@mcp.tool()
async def clear_ssd_cache() -> Any:
    """Clear the on-disk (cold-tier) KV cache. Destructive — frees disk, drops prefix reuse."""
    return await _admin("POST", "/admin/api/ssd-cache/clear")


@mcp.tool()
async def clear_hot_cache() -> Any:
    """Clear the in-memory (hot-tier) KV cache. Destructive — frees RAM, drops recent prefixes."""
    return await _admin("POST", "/admin/api/hot-cache/clear")


# ── Inference ─────────────────────────────────────────────────────────────────


@mcp.tool()
async def chat(
    message: str,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Any:
    """Send one user message and return the assistant reply (POST /v1/chat/completions).

    Requires a model to be available; if list_models is empty, bootstrap with
    search_models + download_model + load_model. Omit `model` to use the default.
    """
    model = model or await _resolve_model()
    if not model:
        return {"error": "NO_MODEL", "message": "No model specified and none available. Use list_models / download_model first."}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": message})
    body: dict[str, Any] = {"messages": messages, "stream": False, "model": model}
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    result = await _v1("POST", "/v1/chat/completions", json=body)
    if isinstance(result, dict) and not result.get("error"):
        choices = result.get("choices") or []
        text = choices[0].get("message", {}).get("content", "") if choices else ""
        return {"response": text, "model": result.get("model"), "usage": result.get("usage")}
    return result


@mcp.tool()
async def complete(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 256,
    temperature: Optional[float] = None,
) -> Any:
    """Raw text completion of a prompt prefix (POST /v1/completions)."""
    model = model or await _resolve_model()
    if not model:
        return {"error": "NO_MODEL", "message": "No model specified and none available. Use list_models / download_model first."}
    body: dict[str, Any] = {"prompt": prompt, "max_tokens": max_tokens, "stream": False, "model": model}
    if temperature is not None:
        body["temperature"] = temperature
    result = await _v1("POST", "/v1/completions", json=body)
    if isinstance(result, dict) and not result.get("error"):
        choices = result.get("choices") or []
        return {"completion": choices[0].get("text", "") if choices else "", "usage": result.get("usage")}
    return result


@mcp.tool()
async def embed(text: str, model: Optional[str] = None) -> Any:
    """Compute an embedding for text (POST /v1/embeddings). Needs an embedding model loaded."""
    body: dict[str, Any] = {"input": text}
    if model:
        body["model"] = model
    return await _v1("POST", "/v1/embeddings", json=body)


if __name__ == "__main__":
    mcp.run()
