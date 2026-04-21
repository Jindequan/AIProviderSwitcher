#!/usr/bin/env python3
"""
Smart Proxy - 多Provider自动切换代理
支持 Claude Code (Anthropic协议) 和 Codex (OpenAI协议)
"""

import os
import json
import time
import asyncio
import re
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, AsyncGenerator, Any
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import anthropic
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn


# ============ 数据模型 ============

@dataclass
class Endpoint:
    protocol: str
    base_url: str
    api_key: str
    retry_at: float = 0.0
    
    def is_available(self) -> bool:
        return time.time() >= self.retry_at
    
    def set_cooldown(self, until: float):
        self.retry_at = max(self.retry_at, until)
        wait_min = (self.retry_at - time.time()) / 60
        print(f"[{self.protocol}] 冷却至 {datetime.fromtimestamp(self.retry_at)} ({wait_min:.1f}min)")


@dataclass
class Provider:
    name: str
    endpoints: Dict[str, Endpoint]
    models: Dict[str, str]
    priority: int
    
    def get_endpoint(self, protocol: str) -> Optional[Endpoint]:
        return self.endpoints.get(protocol)
    
    def is_available(self, protocol: str) -> bool:
        ep = self.get_endpoint(protocol)
        return ep is not None and ep.is_available()
    
    def get_model_name(self, user_model: str) -> Optional[str]:
        if user_model in self.models:
            return self.models[user_model]
        if "default" in self.models:
            print(f"[{self.name}] 模型 {user_model} 未找到，使用 default")
            return self.models["default"]
        return None


# ============ 配置加载 ============

DEFAULT_CONFIG = [
    {
        "name": "ClaudeOfficial",
        "endpoints": {
            "anthropic": "https://api.anthropic.com/v1"
        },
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "models": {
            "default": "claude-3-5-sonnet-20241022"
        },
        "priority": 1
    }
]


def load_config(path: str) -> List[Provider]:
    if not os.path.exists(path):
        print(f"配置文件 {path} 不存在，使用默认配置")
        data = DEFAULT_CONFIG
    else:
        with open(path) as f:
            data = json.load(f)

    if isinstance(data, dict) and "providers" in data:
        data = data.get("providers") or []
    
    providers = []
    for item in data or []:
        if isinstance(item, dict) and item.get("enabled", True) is False:
            continue
        endpoints = {}
        for proto, cfg in item["endpoints"].items():
            if isinstance(cfg, str):
                endpoints[proto] = Endpoint(
                    protocol=proto,
                    base_url=cfg.rstrip("/"),
                    api_key=item.get("api_key", "")
                )
            else:
                endpoints[proto] = Endpoint(
                    protocol=proto,
                    base_url=cfg["base_url"].rstrip("/"),
                    api_key=cfg.get("api_key", item.get("api_key", ""))
                )
        
        providers.append(Provider(
            name=item["name"],
            endpoints=endpoints,
            models=item.get("models", {}),
            priority=item.get("priority", 999)
        ))
    
    providers.sort(key=lambda p: (p.priority, p.name))
    return providers


# ============ 协议工具 ============

def detect_protocol(request: Request) -> str:
    path = request.url.path
    if "/v1/messages" in path:
        return "anthropic"
    return "openai"


ALLOWED_PARAMS = {
    "model", "messages", "max_tokens", "temperature", "stream",
    "system", "tools", "tool_choice", "top_p", "top_k",
    "stop_sequences", "metadata"
}


def clean_body(body: Dict) -> Dict:
    """过滤 Anthropic SDK 不支持的参数"""
    return {k: v for k, v in body.items() if k in ALLOWED_PARAMS}


def openai_request_to_anthropic(body: Dict) -> Dict:
    messages = body.get("messages", [])
    system = None
    anthropic_messages = []
    
    for msg in messages:
        if msg.get("role") == "system":
            system = msg.get("content")
        else:
            anthropic_messages.append({
                "role": msg.get("role"),
                "content": msg.get("content")
            })
    
    result = {
        "model": body.get("model"),
        "messages": anthropic_messages,
        "max_tokens": body.get("max_tokens", 4096),
        "temperature": body.get("temperature", 0),
        "stream": body.get("stream", False),
    }
    
    if system:
        result["system"] = system
    
    if body.get("tools"):
        result["tools"] = body["tools"]
    
    return result


def anthropic_response_to_openai(msg) -> Dict:
    content = ""
    for block in msg.content:
        if hasattr(block, "text"):
            content += block.text
    
    return {
        "id": msg.id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": msg.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop" if msg.stop_reason == "end_turn" else msg.stop_reason
        }],
        "usage": {
            "prompt_tokens": msg.usage.input_tokens if msg.usage else 0,
            "completion_tokens": msg.usage.output_tokens if msg.usage else 0,
            "total_tokens": (msg.usage.input_tokens + msg.usage.output_tokens) if msg.usage else 0
        }
    }


def anthropic_to_openai_sse(event, event_id: str) -> Optional[str]:
    """Anthropic 流式事件转 OpenAI SSE"""
    if event.type == "content_block_delta" and hasattr(event.delta, "text"):
        data = {
            "id": event_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "claude",
            "choices": [{
                "index": 0,
                "delta": {"content": event.delta.text},
                "finish_reason": None
            }]
        }
        return f"data: {json.dumps(data)}\n\n"
    elif event.type == "message_stop":
        data = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        return f"data: {json.dumps(data)}\n\n"
    return None


def _anthropic_event_to_sse_bytes(event: Any) -> bytes:
    event_type = getattr(event, "type", None) or "event"
    payload: Dict[str, Any] = {"type": event_type}

    if event_type == "content_block_delta":
        delta = getattr(event, "delta", None)
        if delta is not None:
            delta_type = getattr(delta, "type", None)
            if delta_type is not None:
                payload["delta"] = {"type": delta_type}
            text = getattr(delta, "text", None)
            if text is not None:
                payload.setdefault("delta", {})["text"] = text

    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {data}\n\n".encode("utf-8")


def _openai_chat_completions_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return ""
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _openai_headers(api_key: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = (api_key or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _anthropic_to_openai_body(body: Dict) -> Dict:
    """Anthropic /v1/messages 请求体转 OpenAI chat 格式。"""
    messages: List[Dict] = []
    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text = "\n".join(b.get("text", "") for b in system if b.get("type") == "text")
            if text:
                messages.append({"role": "system", "content": text})
    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            messages.append({"role": role, "content": text})
    oai: Dict = {"model": body.get("model", ""), "messages": messages}
    for src, dst in [("max_tokens", "max_tokens"), ("temperature", "temperature"),
                     ("top_p", "top_p"), ("stream", "stream")]:
        if src in body:
            oai[dst] = body[src]
    if "stop_sequences" in body:
        oai["stop"] = body["stop_sequences"]
    return oai


def _openai_resp_to_anthropic_dict(oai: Dict, original_model: str) -> Dict:
    """非流式 OpenAI 响应转 Anthropic message 格式。"""
    choice = (oai.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    finish = choice.get("finish_reason") or "stop"
    usage = oai.get("usage") or {}
    return {
        "id": oai.get("id", "msg_unknown"),
        "type": "message",
        "role": "assistant",
        "model": original_model,
        "content": [{"type": "text", "text": content}],
        "stop_reason": "max_tokens" if finish == "length" else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


async def _openai_sse_to_anthropic_bytes(
    resp: Any, msg_id: str, model: str
) -> AsyncGenerator[bytes, None]:
    """将 OpenAI SSE 流转换为 Anthropic SSE 格式输出。"""
    def evt(event: str, data: Any) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    yield evt("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": model,
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 1},
        },
    })
    yield evt("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    })
    yield b'event: ping\ndata: {"type":"ping"}\n\n'

    finish_reason = "end_turn"
    output_tokens = 0

    async for line in resp.aiter_lines():
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            fr = choices[0].get("finish_reason")
            if fr:
                finish_reason = "max_tokens" if fr == "length" else "end_turn"
            text = delta.get("content")
            if text:
                output_tokens += 1
                yield evt("content_block_delta", {
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": text},
                })
        except (json.JSONDecodeError, KeyError):
            continue

    yield evt("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield evt("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": finish_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'


def _parse_datetime_to_epoch(value: str) -> Optional[float]:
    s = (value or "").strip().strip('"').strip("'")
    if not s:
        return None

    upper = s.upper()
    if upper.endswith(" UTC") or upper.endswith(" GMT"):
        s = s.rsplit(" ", 1)[0].strip()
        try:
            dt = datetime.fromisoformat(s.replace(" ", "T", 1))
        except Exception:
            dt = None
        if dt is None:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f"):
                try:
                    dt = datetime.strptime(s, fmt)
                    break
                except Exception:
                    dt = None
        if dt is None:
            return None
        return dt.replace(tzinfo=timezone.utc).timestamp()

    if re.search(r"[+-]\d{4}$", s):
        s = s[:-5] + s[-5:-2] + ":" + s[-2:]
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    candidates = [s]
    if " " in s and "T" not in s:
        candidates.append(s.replace(" ", "T", 1))

    for cand in candidates:
        try:
            dt = datetime.fromisoformat(cand)
            if dt.tzinfo is None:
                return dt.timestamp()
            return dt.timestamp()
        except Exception:
            pass

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.timestamp()
        except Exception:
            pass

    return None


def _extract_duration_hint_seconds(text: str) -> Optional[int]:
    if not text:
        return None

    patterns = [
        (r"(?:in|for)\s+(\d+)\s*(?:hours?|hrs?|hr|h)\b", 3600),
        (r"(?:in|for)\s+(\d+)\s*(?:minutes?|mins?|min|m)\b", 60),
        (r"(?:in|for)\s+(\d+)\s*(?:seconds?|secs?|sec|s)\b", 1),
    ]
    for pattern, scale in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1)) * scale
            except Exception:
                return None
    return None


def _parse_reset_at_epoch(value: str, *, now: float, duration_hint_seconds: Optional[int]) -> Optional[float]:
    s = (value or "").strip().strip('"').strip("'")
    if not s:
        return None

    has_tz = bool(re.search(r"(Z|[+-]\d{2}:?\d{2})$", s)) or s.upper().endswith(" UTC") or s.upper().endswith(" GMT")
    if has_tz:
        return _parse_datetime_to_epoch(s)

    cand = s.replace(" ", "T", 1) if " " in s and "T" not in s else s
    dt: Optional[datetime] = None
    try:
        dt = datetime.fromisoformat(cand)
    except Exception:
        dt = None

    if dt is None:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except Exception:
                dt = None

    if dt is None:
        return None

    local_ts = dt.timestamp()
    utc_ts = dt.replace(tzinfo=timezone.utc).timestamp()
    candidates = [t for t in (local_ts, utc_ts) if t >= now]
    if not candidates:
        return None

    if duration_hint_seconds is not None:
        target = now + duration_hint_seconds
        return min(candidates, key=lambda t: abs(t - target))

    return min(candidates)


def _compute_retry_at_epoch(
    status_code: int,
    headers: Dict[str, str],
    error_body: str = "",
    default_fallback: int = 3600
) -> Optional[float]:
    """
    从多个来源解析精确的 retry_at 时间

    优先级:
    1. HTTP 响应头 (Retry-After, X-RateLimit-Reset, etc.)
    2. 响应体 JSON (error.retry_after, error.retry_at)
    3. 响应体文本正则 (until, reset at, in X minutes)
    4. 按错误类型的默认值

    Args:
        status_code: HTTP 状态码
        headers: 响应头字典
        error_body: 响应体内容
        default_fallback: 429 错误的默认冷却时间（秒）

    Returns:
        Unix 时间戳，如果无法解析则返回 None
    """
    now = time.time()

    # === 1. HTTP 响应头 (所有错误都检查) ===
    h = {str(k).lower(): str(v) for k, v in (headers or {}).items()}

    # Retry-After (RFC 7232)
    retry_after = h.get("retry-after")
    if retry_after:
        try:
            if retry_after.isdigit():
                return now + int(retry_after)
            else:
                dt = parsedate_to_datetime(retry_after)
                return dt.timestamp()
        except Exception:
            pass

    # X-RateLimit-* 系列
    for key in ["x-ratelimit-reset", "x-ratelimit-reset-requests",
                "x-ratelimit-reset-tokens", "ratelimit-reset"]:
        val = h.get(key)
        if val:
            try:
                ts = int(val)
                return ts if ts < 10000000000 else ts / 1000
            except Exception:
                pass

    # Cloudflare 特有
    cf_retry = h.get("cf-ray-status-retry-after")
    if cf_retry:
        try:
            return now + int(cf_retry)
        except Exception:
            pass

    # === 2. 响应体 JSON ===
    try:
        error_json = json.loads(error_body)
        if "error" in error_json:
            error = error_json["error"]

            # retry_at (绝对时间)
            if "retry_at" in error:
                duration_hint = _extract_duration_hint_seconds(error_body)
                parsed = _parse_reset_at_epoch(
                    str(error["retry_at"]),
                    now=now,
                    duration_hint_seconds=duration_hint
                )
                if parsed:
                    return parsed

            # retry_after (相对秒数)
            if "retry_after" in error:
                if isinstance(error["retry_after"], (int, float)):
                    return now + float(error["retry_after"])
                if isinstance(error["retry_after"], dict):
                    seconds = error["retry_after"].get("seconds") or \
                              error["retry_after"].get("value")
                    if seconds:
                        return now + float(seconds)
    except Exception:
        pass

    # === 3. 响应体文本正则 ===
    duration_hint = _extract_duration_hint_seconds(error_body or "")

    patterns = [
        # 绝对时间 - Unix 时间戳
        (r'until\s+(\d{10,13})',
         lambda m: int(m.group(1)) / (1000 if len(m.group(1)) > 10 else 1)),

        # 绝对时间 - ISO 8601 (支持时区)
        (r'(?:will\s+)?reset\s+at\s+(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)',
         lambda m: _parse_reset_at_epoch(m.group(1), now=now, duration_hint_seconds=duration_hint)),

        (r'retry\s+at\s+(\d{4}-\d{2}-\d{2}[^\s,}]+)',
         lambda m: _parse_reset_at_epoch(m.group(1), now=now, duration_hint_seconds=duration_hint)),

        # 相对时间 - 秒
        (r'(?:in|retry|wait)\s+(\d+)\s*s(?:econds?)?',
         lambda m: now + int(m.group(1))),

        (r'retry\s+after\s+(\d+)\s*(?:seconds?|s)',
         lambda m: now + int(m.group(1))),

        # 相对时间 - 分钟
        (r'(?:in|for)\s+(\d+)\s*(?:minutes?|mins?|min|m)\b',
         lambda m: now + int(m.group(1)) * 60),

        # 相对时间 - 小时
        (r'(?:in|for)\s+(\d+)\s*(?:hours?|hrs?|hr|h)\b',
         lambda m: now + int(m.group(1)) * 3600),
    ]

    for pattern, extractor in patterns:
        match = re.search(pattern, error_body or "", re.IGNORECASE)
        if match:
            try:
                retry_at = extractor(match)
                if isinstance(retry_at, (int, float)) and retry_at > now:
                    return float(retry_at)
            except Exception:
                pass

    # === 4. 按错误类型返回默认值 ===
    if status_code == 429:
        return now + default_fallback
    elif status_code >= 500:
        return now + 30
    elif status_code in (401, 403):
        return now + 300
    elif status_code == 408:
        return now + 60
    elif status_code == 400:
        return now + 60

    return None


# ============ 核心路由 ============

class SmartRouter:
    def __init__(self, providers: List[Provider]):
        self.providers = providers
    
    def _eligible_providers(self, protocol: str, model: str) -> List[Provider]:
        eligible = []
        for p in self.providers:
            ep = p.get_endpoint(protocol)
            # 跨协议回退：anthropic 请求也可路由到 openai provider（会做格式转换）
            if ep is None and protocol == "anthropic":
                ep = p.get_endpoint("openai")
            if ep is None:
                continue
            if not ep.is_available():
                continue
            if p.get_model_name(model):
                eligible.append(p)
        return sorted(eligible, key=lambda p: p.priority)

    def _build_no_provider_error(
        self,
        *,
        protocol: str,
        model: str,
        tried_providers: List[str],
        last_error: Optional[Exception],
    ) -> str:
        now = time.time()
        candidates: List[Provider] = []
        for p in self.providers:
            ep = p.get_endpoint(protocol)
            if ep is None and protocol == "anthropic":
                ep = p.get_endpoint("openai")
            if ep is None:
                continue
            if not p.get_model_name(model):
                continue
            candidates.append(p)

        if not candidates:
            base = f"所有 provider 都不可用（无 provider 支持 protocol={protocol}, model={model}）"
            if last_error is not None:
                return f"{base}，最后错误: {last_error}"
            return base

        soonest_provider: Optional[Provider] = None
        soonest_time: Optional[float] = None
        for p in candidates:
            ep = p.get_endpoint(protocol)
            if ep is None and protocol == "anthropic":
                ep = p.get_endpoint("openai")
            if ep is None:
                continue
            if soonest_time is None or ep.retry_at < soonest_time:
                soonest_time = ep.retry_at
                soonest_provider = p

        def fmt(ts: Optional[float]) -> str:
            if ts is None:
                return "unknown"
            if ts == float("inf"):
                return "∞"
            if ts <= 0:
                return "now"
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

        def wait(ts: Optional[float]) -> Optional[int]:
            if ts is None or ts == float("inf"):
                return None
            if ts <= 0:
                return 0
            delta = ts - now
            return int(delta) if delta > 0 else 0

        next_part = ""
        if soonest_provider is not None and soonest_time is not None:
            w = wait(soonest_time)
            if w is None:
                next_part = f"下一次可用: {soonest_provider.name} @ {fmt(soonest_time)}"
            else:
                next_part = f"下一次可用: {soonest_provider.name} @ {fmt(soonest_time)} (in {w}s)"

        if last_error is not None and next_part:
            return f"所有 provider 都不可用，{next_part}，最后错误: {last_error}"
        if last_error is not None:
            return f"所有 provider 都不可用，最后错误: {last_error}"
        if next_part:
            return f"所有 provider 都不可用，{next_part}"
        return "所有 provider 都不可用"

    def _next_available(self, *, protocol: str, model: str) -> Dict[str, Any]:
        now = time.time()
        candidates: List[Provider] = []
        for p in self.providers:
            ep = p.get_endpoint(protocol)
            if ep is None and protocol == "anthropic":
                ep = p.get_endpoint("openai")
            if ep is None:
                continue
            if not p.get_model_name(model):
                continue
            candidates.append(p)

        if not candidates:
            return {"supported": False, "provider": None, "retry_at": None, "retry_after": None}

        soonest_provider: Optional[Provider] = None
        soonest_time: Optional[float] = None
        for p in candidates:
            ep = p.get_endpoint(protocol)
            if ep is None and protocol == "anthropic":
                ep = p.get_endpoint("openai")
            if ep is None:
                continue
            if soonest_time is None or ep.retry_at < soonest_time:
                soonest_time = ep.retry_at
                soonest_provider = p

        retry_after: Optional[int] = None
        if soonest_time is not None and soonest_time != float("inf") and soonest_time > now:
            retry_after = max(1, int(soonest_time - now))

        return {
            "supported": True,
            "provider": soonest_provider.name if soonest_provider is not None else None,
            "retry_at": soonest_time,
            "retry_after": retry_after,
        }
    
    async def _try_provider_stream(
        self,
        provider: Provider,
        protocol: str,
        model: str,
        body: Dict
    ) -> AsyncGenerator[bytes, None]:
        """尝试单个 provider 的流式请求，成功则 yield 数据，失败则 raise。

        支持跨协议：anthropic 请求可路由到 openai endpoint（自动格式转换）。
        """
        endpoint = provider.get_endpoint(protocol)
        actual_protocol = protocol
        needs_translation = False

        if endpoint is None and protocol == "anthropic":
            endpoint = provider.get_endpoint("openai")
            if endpoint is not None:
                actual_protocol = "openai"
                needs_translation = True

        if endpoint is None:
            raise Exception(f"Provider {provider.name} 不支持 {protocol}")

        actual_model = provider.get_model_name(model)
        if not actual_model:
            raise Exception(f"Provider {provider.name} 没有模型 {model}")

        body["model"] = actual_model

        try:
            if actual_protocol == "openai":
                send_body = _anthropic_to_openai_body(body) if needs_translation else body
                url = _openai_chat_completions_url(endpoint.base_url)
                if not url:
                    raise Exception(f"Provider {provider.name} openai base_url 为空")
                async with httpx.AsyncClient(timeout=60.0) as client:
                    async with client.stream(
                        "POST",
                        url,
                        json=send_body,
                        headers=_openai_headers(endpoint.api_key),
                    ) as resp:
                        if resp.status_code >= 400:
                            raw = await resp.aread()
                            text = raw.decode(errors="ignore")
                            retry_at = _compute_retry_at_epoch(resp.status_code, dict(resp.headers), text)
                            if retry_at:
                                endpoint.set_cooldown(retry_at)
                            raise Exception(f"HTTP {resp.status_code}: {text[:200]}")
                        if needs_translation:
                            msg_id = resp.headers.get("x-request-id", f"msg_{provider.name[:6]}")
                            async for chunk in _openai_sse_to_anthropic_bytes(resp, msg_id, model):
                                yield chunk
                        else:
                            async for chunk in resp.aiter_raw():
                                yield chunk
                return

            body = clean_body(body)
            client = anthropic.AsyncAnthropic(api_key=endpoint.api_key, base_url=endpoint.base_url)
            params = dict(body)
            params["stream"] = True
            stream = await client.messages.create(**params)
            async for event in stream:
                yield _anthropic_event_to_sse_bytes(event)

        except anthropic.RateLimitError as e:
            headers: Dict = {}
            body_text = str(e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    headers = dict(e.response.headers)
                except Exception:
                    pass
                try:
                    body_text = e.response.content.decode(errors="ignore")
                except Exception:
                    pass
            retry_at = _compute_retry_at_epoch(429, headers, body_text)
            if retry_at:
                endpoint.set_cooldown(retry_at)
            raise
        except anthropic.AuthenticationError as e:
            headers = {}
            body_text = str(e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    headers = dict(e.response.headers)
                except Exception:
                    pass
                try:
                    body_text = e.response.content.decode(errors="ignore")
                except Exception:
                    pass
            retry_at = _compute_retry_at_epoch(401, headers, body_text)
            if retry_at:
                endpoint.set_cooldown(retry_at)
            raise
        except anthropic.APIStatusError as e:
            headers = {}
            body_text = str(e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    headers = dict(e.response.headers)
                except Exception:
                    pass
                try:
                    body_text = e.response.content.decode(errors="ignore")
                except Exception:
                    pass
            retry_at = _compute_retry_at_epoch(e.status_code, headers, body_text)
            if retry_at:
                endpoint.set_cooldown(retry_at)
            raise
        except Exception as e:
            error_str = str(e)
            # SSE error events (HTTP 200 with event:error body) or other unhandled errors
            retry_at = _compute_retry_at_epoch(400, {}, error_str)
            if retry_at and endpoint.retry_at < retry_at:
                endpoint.set_cooldown(retry_at)
            raise

    async def _try_provider_non_stream(
        self,
        provider: Provider,
        protocol: str,
        model: str,
        body: Dict
    ) -> Dict:
        """尝试单个 provider 的非流式请求，支持跨协议格式转换。"""
        endpoint = provider.get_endpoint(protocol)
        actual_protocol = protocol
        needs_translation = False

        if endpoint is None and protocol == "anthropic":
            endpoint = provider.get_endpoint("openai")
            if endpoint is not None:
                actual_protocol = "openai"
                needs_translation = True

        if endpoint is None:
            raise Exception(f"Provider {provider.name} 不支持 {protocol}")

        actual_model = provider.get_model_name(model)
        if not actual_model:
            raise Exception(f"Provider {provider.name} 没有模型 {model}")

        body["model"] = actual_model

        try:
            if actual_protocol == "openai":
                send_body = _anthropic_to_openai_body(body) if needs_translation else body
                url = _openai_chat_completions_url(endpoint.base_url)
                if not url:
                    raise Exception(f"Provider {provider.name} openai base_url 为空")
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(url, json=send_body, headers=_openai_headers(endpoint.api_key))
                if resp.status_code >= 400:
                    retry_at = _compute_retry_at_epoch(resp.status_code, dict(resp.headers), resp.text)
                    if retry_at:
                        endpoint.set_cooldown(retry_at)
                    raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
                oai_json = resp.json()
                return _openai_resp_to_anthropic_dict(oai_json, model) if needs_translation else oai_json

            body = clean_body(body)
            client = anthropic.AsyncAnthropic(api_key=endpoint.api_key, base_url=endpoint.base_url)
            response = await client.messages.create(**body)
            if not hasattr(response, "model_dump"):
                return response
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return response.model_dump()

        except anthropic.RateLimitError as e:
            headers: Dict = {}
            body_text = str(e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    headers = dict(e.response.headers)
                except Exception:
                    pass
                try:
                    body_text = e.response.content.decode(errors="ignore")
                except Exception:
                    pass
            retry_at = _compute_retry_at_epoch(429, headers, body_text)
            if retry_at:
                endpoint.set_cooldown(retry_at)
            raise
        except anthropic.AuthenticationError as e:
            headers = {}
            body_text = str(e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    headers = dict(e.response.headers)
                except Exception:
                    pass
                try:
                    body_text = e.response.content.decode(errors="ignore")
                except Exception:
                    pass
            retry_at = _compute_retry_at_epoch(401, headers, body_text)
            if retry_at:
                endpoint.set_cooldown(retry_at)
            raise
        except anthropic.APIStatusError as e:
            headers = {}
            body_text = str(e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    headers = dict(e.response.headers)
                except Exception:
                    pass
                try:
                    body_text = e.response.content.decode(errors="ignore")
                except Exception:
                    pass
            retry_at = _compute_retry_at_epoch(e.status_code, headers, body_text)
            if retry_at:
                endpoint.set_cooldown(retry_at)
            raise
        except Exception as e:
            error_str = str(e)
            retry_at = _compute_retry_at_epoch(400, {}, error_str)
            if retry_at and endpoint.retry_at < retry_at:
                endpoint.set_cooldown(retry_at)
            raise

    async def request_stream(
        self,
        protocol: str,
        model: str,
        body: Dict
    ) -> AsyncGenerator[bytes, None]:
        """流式请求，自动 failover"""
        last_error = None
        tried_providers = []
        last_provider: Optional[str] = None
        
        for attempt in range(len(self.providers) * 2):
            eligible = self._eligible_providers(protocol, model)
            # 排除已尝试的
            eligible = [p for p in eligible if p.name not in tried_providers]
            
            if not eligible:
                # 全部冷却或已尝试，找最快恢复的
                soonest_time = float('inf')
                soonest_provider = None
                
                for p in self.providers:
                    if p.name in tried_providers:
                        continue
                    ep = p.get_endpoint(protocol)
                    if ep and ep.retry_at < soonest_time:
                        soonest_time = ep.retry_at
                        soonest_provider = p
                
                if soonest_provider:
                    wait = soonest_time - time.time()
                    if 0 < wait < 300:
                        print(f"全部冷却，等待 {wait:.0f} 秒...")
                        await asyncio.sleep(wait)
                        eligible = [soonest_provider]
                    else:
                        break
            
            for provider in eligible:
                tried_providers.append(provider.name)
                if last_error is not None and last_provider is not None:
                    print(f"[SWITCH] {last_provider} -> {provider.name} ({type(last_error).__name__}: {str(last_error)[:120]})")
                try:
                    async for chunk in self._try_provider_stream(provider, protocol, model, body.copy()):
                        yield chunk
                    return  # 成功完成
                except Exception as e:
                    last_error = e
                    last_provider = provider.name
                    continue
            
            await asyncio.sleep(0.1)
        
        # 全部失败
        error_msg = self._build_no_provider_error(
            protocol=protocol,
            model=model,
            tried_providers=tried_providers,
            last_error=last_error,
        )
        print(error_msg)
        yield f"data: {json.dumps({'error': error_msg}, ensure_ascii=False)}\n\n".encode("utf-8")
    
    async def request_non_stream(
        self,
        protocol: str,
        model: str,
        body: Dict
    ) -> Dict:
        """非流式请求，自动 failover"""
        last_error = None
        tried_providers = []
        last_provider: Optional[str] = None
        
        for attempt in range(len(self.providers) * 2):
            eligible = self._eligible_providers(protocol, model)
            eligible = [p for p in eligible if p.name not in tried_providers]
            
            if not eligible:
                soonest_time = float('inf')
                soonest_provider = None
                
                for p in self.providers:
                    if p.name in tried_providers:
                        continue
                    ep = p.get_endpoint(protocol)
                    if ep and ep.retry_at < soonest_time:
                        soonest_time = ep.retry_at
                        soonest_provider = p
                
                if soonest_provider:
                    wait = soonest_time - time.time()
                    if 0 < wait < 300:
                        print(f"全部冷却，等待 {wait:.0f} 秒...")
                        await asyncio.sleep(wait)
                        eligible = [soonest_provider]
                    else:
                        break
            
            for provider in eligible:
                tried_providers.append(provider.name)
                if last_error is not None and last_provider is not None:
                    print(f"[SWITCH] {last_provider} -> {provider.name} ({type(last_error).__name__}: {str(last_error)[:120]})")
                try:
                    return await self._try_provider_non_stream(provider, protocol, model, body.copy())
                except Exception as e:
                    last_error = e
                    last_provider = provider.name
                    continue
            
            await asyncio.sleep(0.1)
        
        msg = self._build_no_provider_error(
            protocol=protocol,
            model=model,
            tried_providers=tried_providers,
            last_error=last_error,
        )
        pre = self._next_available(protocol=protocol, model=model)
        if pre.get("supported") and pre.get("retry_after") is not None:
            raise NoProviderAvailable(msg, retry_after=int(pre["retry_after"]))
        raise Exception(msg)


# ============ FastAPI应用 ============

router: Optional[SmartRouter] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global router
    config_path = os.getenv("PROXY_CONFIG", "config.json")
    providers = load_config(config_path)
    router = SmartRouter(providers)
    print(f"加载了 {len(providers)} 个Provider: {[p.name for p in providers]}")
    yield
    print("Shutdown")


app = FastAPI(title="Smart Proxy", lifespan=lifespan)


class NoProviderAvailable(Exception):
    def __init__(self, message: str, *, retry_after: Optional[int]):
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after


@app.post("/v1/messages")
@app.post("/v1/chat/completions")
async def proxy(request: Request):
    protocol = detect_protocol(request)
    body = await request.json()
    
    model = body.get("model", "claude-sonnet")
    accept = (request.headers.get("accept") or "").lower()
    stream = bool(body.get("stream", False)) or ("text/event-stream" in accept)
    
    try:
        pre = router._next_available(protocol=protocol, model=model)
        if pre.get("supported") and pre.get("retry_after") is not None:
            msg = router._build_no_provider_error(
                protocol=protocol,
                model=model,
                tried_providers=[],
                last_error=None,
            )
            raise NoProviderAvailable(msg, retry_after=int(pre["retry_after"]))

        if stream:
            body = dict(body)
            body["stream"] = True
            return StreamingResponse(
                router.request_stream(protocol, model, body),
                media_type="text/event-stream",
            )
        else:
            result = await router.request_non_stream(protocol, model, body)
            return JSONResponse(content=result)
            
    except NoProviderAvailable as e:
        headers = {"Retry-After": str(e.retry_after)} if e.retry_after is not None else None
        raise HTTPException(status_code=429, detail=e.message, headers=headers)
    except Exception as e:
        print(f"最终失败: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/health")
def health():
    status = {}
    for p in router.providers:
        for proto, ep in p.endpoints.items():
            key = f"{p.name}/{proto}"
            available = ep.is_available()
            status[key] = {
                "available": available,
                "retry_at": datetime.fromtimestamp(ep.retry_at).isoformat() if ep.retry_at > 0 else None,
                "wait_seconds": max(0, ep.retry_at - time.time()) if not available else 0
            }
    return status


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888, access_log=False)
