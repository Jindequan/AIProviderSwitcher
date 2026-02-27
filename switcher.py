import json
import time
import datetime
import requests
import uvicorn
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_config()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "ok", "providers": len(PROVIDERS), "cooldowns": len(COOLDOWNS)}

CONFIG_FILE = "config.json"
GLOBAL_CONFIG = {}
PROVIDERS = {}
COOLDOWNS: Dict[str, float] = {}  # provider_name -> timestamp

def load_config():
    global GLOBAL_CONFIG, PROVIDERS
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            GLOBAL_CONFIG = json.load(f)
            # Filter enabled providers and sort by priority
            raw_providers = [p for p in GLOBAL_CONFIG.get("providers", []) if p.get("enabled", True)]
            PROVIDERS = {p["name"]: p for p in raw_providers}
            print(f"Loaded {len(PROVIDERS)} providers: {list(PROVIDERS.keys())}")
    except Exception as e:
        print(f"Error loading config: {e}")

class ProviderManager:
    @staticmethod
    def get_sorted_providers(protocol: str) -> List[Dict[str, Any]]:
        """Return list of providers for the given protocol, sorted by priority."""
        current_time = time.time()
        
        candidates = []
        for name, p in PROVIDERS.items():
            # Check protocol match (default to 'anthropic' if not set for backward compatibility)
            p_proto = p.get("protocol", "anthropic").lower()
            if p_proto != protocol.lower():
                continue

            # Check cooldown
            cooldown_until = COOLDOWNS.get(name, 0)
            if current_time < cooldown_until:
                # print(f"Provider {name} is cooling down until {datetime.datetime.fromtimestamp(cooldown_until)}")
                continue
            candidates.append(p)
        
        # Sort by priority
        candidates.sort(key=lambda x: x.get("priority", 999))
        return candidates

    @staticmethod
    def mark_success(provider_name: str):
        if provider_name in COOLDOWNS:
            del COOLDOWNS[provider_name]

    @staticmethod
    def mark_failure(provider_name: str, status_code: int, retry_after: Optional[int] = None):
        failover_cfg = GLOBAL_CONFIG.get("failover", {})
        default_cooldown = failover_cfg.get("cooldown_seconds", 60)
        cooldown_429 = failover_cfg.get("cooldown_429_seconds", 300)
        cooldown_403 = failover_cfg.get("cooldown_403_seconds", 3600)
        
        duration = default_cooldown
        if status_code == 429:
            duration = retry_after if retry_after else cooldown_429
        elif status_code == 403 or status_code == 401:
            duration = cooldown_403
        
        COOLDOWNS[provider_name] = time.time() + duration
        print(f"Provider {provider_name} failed (status {status_code}). Cooldown for {duration}s.")

import re

def map_model(provider: Dict[str, Any], original_model: str) -> str:
    # 1. 如果配置了 models 映射表，优先查找
    models_map = provider.get("models", {})
    if original_model in models_map:
        return models_map[original_model]
    
    # 2. 尝试根据关键词猜测
    for key in ["haiku", "sonnet", "opus", "gpt-3.5", "gpt-4"]:
        if key in original_model.lower() and key in models_map:
            return models_map[key]

    # 3. 如果没找到映射，但 provider 配置了 default_model，则使用默认模型
    # 用户需求：模型可以不配置，不配置走默认 model
    if provider.get("default_model"):
        return provider["default_model"]
            
    # 4. 如果连 default_model 都没有，才透传原始模型（作为最后手段）
    return original_model

def extract_wait_time(text: str) -> Optional[int]:
    """Extract wait time in seconds from error message text."""
    # Match patterns like "try again in 20s", "retry after 30 seconds", "limit resets in 45s"
    # Prioritize seconds
    match = re.search(r'(\d+)\s*(s|sec|second)', text.lower())
    if match:
        return int(match.group(1))
    
    # Sometimes it might be minutes
    match = re.search(r'(\d+)\s*(m|min|minute)', text.lower())
    if match:
        return int(match.group(1)) * 60
        
    return None

def make_headers(provider: Dict[str, Any], incoming_headers: Dict[str, str], protocol: str) -> Dict[str, str]:
    h = {}
    # Pass through most headers
    for k, v in incoming_headers.items():
        lk = k.lower()
        if lk in ("host", "authorization", "x-api-key", "content-length", "connection", "upgrade", "accept-encoding"):
            continue
        h[k] = v
    
    # Set provider specific headers based on protocol
    if protocol == "anthropic":
        h["x-api-key"] = provider["api_key"]
        h["anthropic-version"] = "2023-06-01"
    elif protocol == "openai":
        h["Authorization"] = f"Bearer {provider['api_key']}"
    
    h["content-type"] = "application/json"
    return h

async def proxy_request(request: Request, body: Dict[str, Any], protocol: str, target_path: str):
    candidates = ProviderManager.get_sorted_providers(protocol)
    if not candidates:
        return JSONResponse(status_code=503, content={
            "error": {"message": f"No available providers for protocol '{protocol}' (all cooling down or disabled)."}
        })

    last_error_response = None
    
    for provider in candidates:
        p_name = provider["name"]
        
        # Construct target URL
        base_url = provider["base_url"].rstrip("/")
        # If base_url already contains /v1/..., handle it carefully
        if "/v1" in base_url and target_path.startswith("/v1"):
             # Simple heuristic: if base_url ends with /v1/messages, use it directly?
             # Better: assume base_url is root API endpoint (e.g. https://api.openai.com)
             # But current config has full path for Anthropic. Let's adjust.
             if protocol == "anthropic" and base_url.endswith("/v1/messages"):
                 target_url = base_url # Use as is
             else:
                 target_url = base_url + target_path
        else:
             target_url = base_url + target_path

        # Map model
        original_model = body.get("model", "")
        mapped_model = map_model(provider, original_model)
        new_body = body.copy()
        new_body["model"] = mapped_model
        
        headers = make_headers(provider, dict(request.headers), protocol)
        
        print(f"[APS] {protocol.upper()} -> {p_name} | {original_model} -> {mapped_model}")

        try:
            # Respect client's stream preference
            is_stream = new_body.get("stream", True)
            
            resp = requests.post(
                target_url,
                headers=headers,
                json=new_body,
                stream=is_stream,
                timeout=provider.get("timeout", 60),
                proxies={"http": None, "https": None} 
            )
            
            if resp.status_code == 200:
                ProviderManager.mark_success(p_name)
                
                response_headers = {}
                for k, v in resp.headers.items():
                    if k.lower() not in ("content-encoding", "transfer-encoding", "connection", "content-length"):
                        response_headers[k] = v
                
                return StreamingResponse(
                    resp.iter_content(chunk_size=None),
                    status_code=resp.status_code,
                    headers=response_headers,
                    media_type=resp.headers.get("content-type")
                )
            
            # Special handling for 400 (Bad Request) - check if it's a model error
            if resp.status_code == 400:
                try:
                    # Read error content (it's usually small JSON)
                    error_content = resp.json()
                    error_str = str(error_content).lower()
                    # Keywords indicating model unavailability or invalid request format specific to provider
                    if "model" in error_str or "not found" in error_str or "support" in error_str:
                        print(f"Provider {p_name} 400 Error (Model/Support issue): {error_str[:100]}... Trying next.")
                        # Mark failure with short cooldown as it might be model specific
                        ProviderManager.mark_failure(p_name, 400, retry_after=5) 
                        last_error_response = resp
                        continue
                except:
                    pass

            # Failover Logic
            failover_cfg = GLOBAL_CONFIG.get("failover", {})
            retry_status = failover_cfg.get("retry_on_status", [401, 403, 408, 429, 500, 502, 503, 504])
            
            if resp.status_code in retry_status:
                retry_after = None
                
                # 1. Try Retry-After header
                if "Retry-After" in resp.headers:
                    try:
                        retry_after = int(resp.headers["Retry-After"])
                    except:
                        pass
                
                # 2. Try to extract from body for 429 errors
                if not retry_after and resp.status_code == 429:
                    try:
                        error_text = resp.text
                        extracted_time = extract_wait_time(error_text)
                        if extracted_time:
                            retry_after = extracted_time
                            print(f"Extracted retry time from body: {retry_after}s")
                    except:
                        pass

                ProviderManager.mark_failure(p_name, resp.status_code, retry_after)
                last_error_response = resp
                print(f"Provider {p_name} failed with {resp.status_code}, trying next...")
                continue
            
            else:
                print(f"Provider {p_name} returned non-retriable error {resp.status_code}")
                return StreamingResponse(
                    resp.iter_content(chunk_size=None),
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type")
                )

        except Exception as e:
            print(f"Provider {p_name} exception: {e}")
            ProviderManager.mark_failure(p_name, 999)
            continue

    if last_error_response:
        return JSONResponse(
            status_code=last_error_response.status_code,
            content={"error": {"message": f"All providers failed. Last error: {last_error_response.status_code}"}}
        )
    
    return JSONResponse(status_code=503, content={"error": {"message": "All providers failed with connection errors."}})

# Anthropic Endpoint
@app.post("/v1/messages")
async def handle_anthropic(request: Request):
    try:
        body = await request.json()
    except:
        body = {}
    return await proxy_request(request, body, "anthropic", "/v1/messages")

# Handle count_tokens (return dummy response to silence 404 logs)
@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    return JSONResponse(content={"input_tokens": 0})

# OpenAI Endpoint
@app.post("/v1/chat/completions")
async def handle_openai(request: Request):
    try:
        body = await request.json()
    except:
        body = {}
    return await proxy_request(request, body, "openai", "/v1/chat/completions")

# Fallback for other paths (assume Anthropic for backward compatibility if path matches)
@app.post("/{full_path:path}")
async def handler(request: Request, full_path: str):
    if full_path == "v1/messages":
        return await handle_anthropic(request)
    if full_path == "v1/chat/completions":
        return await handle_openai(request)
    
    # Default behavior?
    return JSONResponse(status_code=404, content={"error": "Not found"})

if __name__ == "__main__":
    load_config()
    host = GLOBAL_CONFIG.get("server", {}).get("host", "127.0.0.1")
    port = GLOBAL_CONFIG.get("server", {}).get("port", 8888)
    uvicorn.run(app, host=host, port=port)
