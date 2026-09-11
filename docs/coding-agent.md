# Local coding agent

`examples/coding_agent.py` runs Qwen or Muse with file tools and a sandboxed
`run_command` tool. It reuses the chat example, tokenizer tool formats, and runtime
decoder. The model receives command output, including exceptions and exit codes,
and can edit and rerun its files. A request is limited to sixteen tool rounds
(configurable with `--max-tool-rounds`).

Use a dedicated folder containing only files the agent may change:

```bash
.venv/bin/python examples/coding_agent.py /path/to/Qwen3.8-27B-GGUF \
  --trusted-folder ./agent-workspace --cache-capacity 16384 \
  --reasoning-effort medium

.venv/bin/python examples/coding_agent.py /path/to/Muse-Glimmer-30B-GGUF \
  --trusted-folder ./agent-workspace --cache-capacity 16384
```

Add `--dflash-path /path/to/assistant` for DFlash acceleration. Qwen also supports
`--mtp` and `--no-thinking`. Sampling defaults are selected by the tokenizer:

| Model / mode | Temperature | Top-p | Top-k | Presence penalty |
|---|---:|---:|---:|---:|
| Qwen, thinking | 1.0 | 0.95 | 20 | 0.0 |
| Qwen, non-thinking | 0.7 | 0.8 | 20 | 1.5 |
| Muse | 1.0 | 0.95 | 64 | 0.0 |

These follow the [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices)
and [Muse](https://huggingface.co/meta-models/Muse-Glimmer-30B) model cards.
Qwen's chat example selects medium reasoning unless another effort is requested.
The resolved sampling settings are printed at startup; CLI overrides remain
available.

For one unattended request, add:

```bash
--prompt 'Write minesweeper.py with a tkinter GUI and a --self-test mode that tests game logic without opening a window. Run python3 -m py_compile minesweeper.py and python3 minesweeper.py --self-test. Fix any errors before finishing.'
```

For a small repair demonstration, put `answer = 42; print(answr)` in `broken.py`
in the workspace and ask: “Run broken.py, inspect the exception, fix it, and run
it again. It should print 42.” The same interaction works with both models.

## Sandbox boundary

The example requires Linux on x86-64 or AArch64, Landlock ABI 6 or newer, and
the system libseccomp library. It rejects `--unsafe-shell`; setup errors never execute the requested
command without confinement. The ordinary chat example can still offer just
file tools on unsupported hosts.

Commands can read file contents in the trusted folder and the system/Python
runtime installation. Writes are confined to the trusted folder. Network sockets,
host display access, private host-file reads, filesystem metadata mutations, and
cross-process control are denied. Landlock enforces filesystem rules; seccomp
uses a default-deny syscall policy because Landlock alone does not cover all of
these operations. See the [kernel Landlock documentation](https://docs.kernel.org/userspace-api/landlock.html).

The helper starts with a clean environment, closed inherited descriptors, no
Python site customization, and an immutable snapshot of its trusted source.
Commands have a wall timeout, a 64 KiB combined-output cap, and per-process CPU,
address-space, file-size, and descriptor limits. Process creation is bounded
relative to the user's existing task count. All descendants remain in the command's
process group, which is killed on success, failure, timeout, cancellation, or
output overflow. Escape cancels a running command as well as generation.

This is local process containment, not a VM: it shares the host kernel and has
no aggregate cgroup memory/disk quota. Files already placed in the trusted folder
are deliberately accessible; do not populate it with private files or hard links
to files you want to protect. Models can overwrite files in that folder. Use
headless self-tests for GUI logic; the sandbox does not expose your desktop.

## Runtime components

- `runtime/sampling.py` applies temperature, top-k, nucleus sampling, and presence
  penalties. Runners optionally provide bounded GPU top-k reads.
- `runtime/chat.py` owns the shared ordinary/MTP/DFlash token iterator, a small
  reasoning-stream parser, and conversation encoding/cache helpers.
- Tokenizers own model-specific chat templates, sampling defaults, and tool-call
  parsing. Reasoning presentation never modifies inference state.
- Runners retain model weights, attention/recurrent state, and GPU execution.
- `runtime/services/coding_tools.py` and `sandbox.py` own application tool access
  and subprocess containment.

This keeps the token policy out of model kernels and avoids adding an engine or
plugin framework. The repetition guard remains a last-resort stop, not a substitute
for correct inference.
