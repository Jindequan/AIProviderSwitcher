# AI Provider Switcher (APS)

[中文文档](README_ZH.md)

APS (AI Provider Switcher) is a universal reverse proxy tool for AI providers, supporting multi-provider automatic switching, failover, and model mapping. It intelligently routes requests to available LLM providers based on configured priorities and health status. This tool is not only suitable for Claude Code CLI but also compatible with any client supporting Anthropic or OpenAI protocols (such as Cursor, VSCode extensions, LangChain, etc.).

## Core Features

*   **Multi-Provider Support**: Configure multiple LLM providers (e.g., Z.AI, Qwen, DeepSeek, OpenAI, etc.), supporting both Anthropic and OpenAI protocols.
*   **Automatic Failover**: Automatically switches to a backup provider when the primary provider encounters 429 (Rate Limit), 5xx (Server Error), or 403 (Auth Error).
*   **Smart Cooldown Mechanism**: Automatically extracts retry times from 429 responses or uses a configured default cooldown time to temporarily disable failed providers.
*   **Model Mapping**: Maps standard model names requested by the client (e.g., `claude-3-haiku`) to provider-specific model names (e.g., `glm-4`, `qwen-max`).
*   **Seamless Switching**: Completely transparent to the client; no need to modify client logic, just point the Base URL to this service.

## Quick Start

### 1. Configure Service

Copy the configuration template:

```bash
cp config.example.json config.json
```

Edit `config.json` and fill in your API Key and Base URL:

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8888,
    "verbose": false
  },
  "providers": [
    {
      "name": "Z_AI",
      "protocol": "anthropic",
      "base_url": "https://api.z.ai/api/anthropic",
      "api_key": "your-key",
      "models": {
        "claude-3-haiku-20240307": "glm-4.7"
      },
      "priority": 1
    }
    // ... more providers
  ]
}
```

### 2. Start Service

**Run directly:**

```bash
python switcher.py
```

**Configure Alias (Zsh):**

Add the following to your `~/.zshrc` file for one-click startup:

```bash
# APS Aliases
# Run in foreground (for viewing logs)
alias aps="python /absolute/path/to/claude-profile-switch/switcher.py"

# Run in background silently
alias aps-start="nohup python /absolute/path/to/claude-profile-switch/switcher.py > /tmp/aps.log 2>&1 & echo 'APS started in background. Logs: /tmp/aps.log'"
alias aps-stop="pkill -f switcher.py && echo 'APS stopped.'"
alias aps-log="tail -f /tmp/aps.log"
```

Remember to replace `/absolute/path/to/...` with your actual code directory path.
Apply configuration: `source ~/.zshrc`

### 3. Configure Client

#### Claude Code CLI

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8888/v1/messages"
```

#### OpenAI Compatible Clients (e.g., LangChain, Cursor)

```bash
export OPENAI_API_BASE="http://127.0.0.1:8888/v1"
# Or configure Base URL in settings
```

Now, all requests will be proxied through APS and automatically distributed according to your configuration.

## Troubleshooting

*   **Logs**: Check `/tmp/aps.log` (if running in background) or terminal output for switching logs.
*   **Port Conflict**: If port 8888 is occupied, change the `port` in `config.json`.
