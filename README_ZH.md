# AI Provider Switcher (APS)

APS 是一个通用的 AI Provider 反向代理工具，支持多 Provider 自动切换、故障转移（Failover）和模型映射。它能够根据配置的 Provider 优先级和健康状态，智能地将请求路由到可用的 LLM 服务商。本工具不仅适用于 Claude Code CLI，也兼容任何支持 Anthropic 或 OpenAI 协议的客户端（如 Cursor, VSCode 插件, LangChain 等）。

## 核心功能

*   **多 Provider 支持**：配置多个 LLM 服务商（如 Z.AI, Qwen, DeepSeek, OpenAI 等），支持 Anthropic 和 OpenAI 协议。
*   **自动故障转移**：当首选 Provider 发生 429 (Rate Limit)、5xx (Server Error) 或 403 (Auth Error) 时，自动无缝切换到备用 Provider。
*   **智能冷却机制**：自动提取 429 响应中的重试时间，或使用配置的默认冷却时间，暂时屏蔽故障 Provider。
*   **模型映射**：将客户端请求的标准模型名称（如 `claude-3-haiku`）映射到 Provider 支持的模型名称（如 `glm-4`、`qwen-max`）。
*   **无感知切换**：对客户端完全透明，无需修改客户端逻辑，只需将 Base URL 指向本服务。

## 快速开始

### 1. 配置服务

复制配置文件模板：

```bash
cp config.example.json config.json
```

编辑 `config.json`，填入你的 API Key 和 Base URL：

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
    // ... 更多 Provider
  ]
}
```

### 2. 启动服务

**直接运行：**

```bash
python switcher.py
```

**配置快捷命令 (Zsh)：**

将以下内容添加到你的 `~/.zshrc` 文件中，实现一键启动：

```bash
# APS 快捷命令
# 前台运行（方便查看日志）
alias aps="python /absolute/path/to/claude-profile-switch/switcher.py"

# 后台静默运行
alias aps-start="nohup python /absolute/path/to/claude-profile-switch/switcher.py > /tmp/aps.log 2>&1 & echo 'APS started in background. Logs: /tmp/aps.log'"
alias aps-stop="pkill -f switcher.py && echo 'APS stopped.'"
alias aps-log="tail -f /tmp/aps.log"
```

记得替换 `/absolute/path/to/...` 为你实际的代码目录路径。
应用配置：`source ~/.zshrc`

### 3. 配置客户端

#### Claude Code CLI

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8888/v1/messages"
```

#### OpenAI 兼容客户端 (如 LangChain, Cursor)

```bash
export OPENAI_API_BASE="http://127.0.0.1:8888/v1"
# 或者在设置中配置 Base URL
```

现在，所有请求都会经过 APS 代理，并根据配置自动分发。

## 故障排查

*   如果遇到 **404 Not Found**，请检查 `base_url` 是否正确，Anthropic 协议通常不需要 `/v1/messages` 后缀（配置中 `base_url` 只需到域名或 `/api/anthropic`，代码会自动拼接，但本工具配置示例中通常包含完整路径，请参考 `config.example.json`）。
*   日志中显示 **Switching** 信息表示触发了故障转移，属于正常现象。
*   **429/403** 错误会自动触发 Provider 冷却，期间该 Provider 不会被选中。

## 许可证

MIT
