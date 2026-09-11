# OpenAI-compatible server and browser chat

## Qwen with DFlash: one command

On this machine, the installed Qwen GGUF and cached DFlash assistant are discovered
automatically:

```bash
.venv/bin/python examples/qwen_dflash_server.py
```

On another installation, give their paths:

```bash
.venv/bin/python examples/qwen_dflash_server.py /path/to/Qwen3.8-27B-GGUF \
  --dflash-path /path/to/Qwen3.8-27B-DFlash2
```

This shortcut uses the shared server implementation. It enables DFlash, listens on
`0.0.0.0:8000`, generates an API key, and defaults to a 32,768-token context with
up to 16,384 generated tokens. No checkpoint downloads happen automatically.
Use `--host 127.0.0.1` for local-only access, `--port` to change the port, or
`--api-key YOUR_SECRET` to retain a fixed key across restarts. Qwen thinking uses
medium effort by default; `--no-thinking` selects the non-thinking sampling defaults.

After loading, startup prints all connection settings, including:

```text
Model ID: qwen3.8-27b
API key: <generated key>
This PC browser: http://127.0.0.1:8000/
This PC API base: http://127.0.0.1:8000/v1
LAN browser: http://192.168.1.42:8000/
LAN API base: http://192.168.1.42:8000/v1
```

The LAN address is detected on your machine; use the actual printed address.
It also prints ready-to-copy terminal-chat and Aider commands for each address.
On machines with multiple network interfaces, the detected preferred address may
need replacing with the address of the interface shared with the client.

## Browser chat

Open the printed browser URL on this PC or another PC on the same network. Paste
the printed API key into the key field and start chatting. The page is served by
the inference server itself: no separate web application, dependencies, account,
CORS setup or browser extension is required. It supports streaming responses,
collapsible thinking, conversation history, new chat and stop. Text/code is shown
as plain text rather than executing model-provided HTML. History and keys remain
in page memory and disappear on reload.

Browser chat is for conversation and code generation. It does not access or edit
repositories. For file edits and command execution, use a coding client.

## Connect a coding agent

For Aider, follow its [official OpenAI-compatible setup instructions](https://aider.chat/docs/llms/openai-compat.html).
Install it **on the client PC**, then run the command printed by the server from
the repository you want to work on. It looks like:

```bash
aider --model openai/qwen3.8-27b \
  --openai-api-base http://192.168.1.42:8000/v1 \
  --openai-api-key YOUR_PRINTED_KEY
```

Use the printed loopback URL when Aider runs on the server PC. Aider's edits and
commands operate on the client PC; the inference server does not execute tool
calls or provide the coding example's sandbox to external clients. Client-specific
approval and execution policies remain the client's responsibility.

Other clients can use the same base URL, model ID and API key if they support
**OpenAI Chat Completions**. This server implements `/v1/chat/completions` (including
streaming and tool-call responses) and `/v1/models`. It does **not** implement
`/v1/responses`, so select a Chat Completions provider rather than a Responses-only
provider. Compatibility with a particular coding agent still depends on the
agent's requested API features and the model's output.

A dependency-free terminal client is also included:

```bash
python examples/openai_client.py --url http://192.168.1.42:8000/v1 \
  --model qwen3.8-27b --api-key YOUR_PRINTED_KEY
```

Copy that single client file to another PC if desired; it requires only Python.

## General server and concurrency

The underlying example supports Qwen, Muse and Nemotron checkpoints:

```bash
.venv/bin/python examples/openai_server.py /path/to/model \
  --host 0.0.0.0 --port 8000 --api-key YOUR_SECRET
```

Add `--dflash-path` for supported Qwen/Muse assistants. DFlash currently uses one
active inference request at a time; additional requests wait. Without DFlash,
`--max-batch-size 1|2|4|8` enables adaptive native continuous batching. Larger
limits reserve more per-request state. DFlash plus multi-request batching is
rejected explicitly instead of silently disabling acceleration.

The general server binds to loopback and has no API key by default. Both launchers
serve the browser page and print connection details. The first request can compile
Warp kernels before generation starts.

LAN clients must be able to reach the selected TCP port; allow it through the
host firewall if necessary. This is a small HTTP server without TLS, intended for
a trusted LAN, not direct Internet exposure. The API key controls inference access
but does not encrypt network traffic.
