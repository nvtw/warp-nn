# Local coding agent

`examples/coding_agent.py` runs Qwen or Muse using the shared chat loop, model
sampling defaults and decoding runtime. It can search a repository, edit files,
run tests, inspect failures and try again. The default budget is 64 tool rounds
per request, with a 32,768-token context; both are configurable.

## Start from a repository

```bash
.venv/bin/python examples/coding_agent.py /path/to/Qwen3.8-27B-GGUF \
  --trusted-folder /path/to/repository --reasoning-effort medium \
  --prompt 'Inspect this project, run its tests, fix the failing tests and validate the changes.'

.venv/bin/python examples/coding_agent.py /path/to/Muse-Glimmer-30B-GGUF \
  --trusted-folder /path/to/repository
```

The example creates a **disposable copy** and prints its session directory,
usually `/tmp/warp-nn-agent-...`. It retains the workspace, an unchanged baseline,
command logs, progress checkpoints and review files. It does not apply changes to
the source repository. On normal exit, inspect `changes.txt`, `changes.patch` and
the edited files in `workspace/`. Review changes before applying selected hunks
or copying files back. Patches describe text changes; binary/large files and empty
file creation may require manual review. The source may have changed since the
copy was made, so check a patch before applying it.

Git metadata, `.agents`, `.codex`, `.env`/`.env.*`, `.ssh`, `.aws`, common dependency
and cache directories are excluded. Symlinks and special files are omitted;
regular hard-linked files are copied into independent files. This is not a secret
scanner: everything else copied from the selected repository is available to the
model. Add exclusions for private files, generated assets or large directories:

```bash
--workspace-exclude 'data/*' --workspace-exclude '*.safetensors'
```

Resume the retained workspace and its progress checkpoint:

```bash
.venv/bin/python examples/coding_agent.py /path/to/model \
  --resume-workspace /tmp/warp-nn-agent-SESSION --prompt 'Continue the remaining work.'
```

Use `/resume` to reopen a saved conversation as well. Command logs and completed
job status survive restart; running processes do not. One session cannot be opened
by two agents simultaneously. Regenerate the review files after an interrupted run:

```bash
.venv/bin/python -m warp_nn.runtime.services.workspace_session /tmp/warp-nn-agent-SESSION
```

Session directories are deliberately retained until you delete them, subject to
your system's temporary-directory cleanup policy.

## Tools and longer tasks

- `list_files` and `search_files` return deterministic pages with explicit next
  offsets. Search supports literal strings, regex (with an installed ripgrep),
  filename globs and up to five surrounding lines. Offsets count output records;
  restart pagination if files change between requests.
- `read_file` returns numbered lines, a SHA256 and a continuation marker. Text
  tools accept files up to 4 MiB; use a bounded command for larger data.
- `write_file` atomically replaces a file. `edit_file` applies one exact replacement
  or several replacements together; if any match fails, nothing is written.
  An optional `expected_sha256` rejects stale edits. This is per-file atomicity,
  not a multi-file transaction.
- `run_command` returns quickly with a job ID for longer commands. Use
  `poll_command`, `read_command_log` (offset or tail), and `cancel_command`.
  Only one command runs at a time; edits wait for it to finish. Bash runs without
  startup files and with `pipefail`, so piping test output does not hide failure.
- `set_executable` safely replaces a compiled artifact with an executable copy.
  Some linkers produce a non-executable output because unrestricted `chmod` is
  blocked. This tool supports regular artifacts up to 256 MiB.
- `save_progress` records the model's task, constraints, discoveries, tests and
  next steps. Context compaction archives the full old history and retains user
  instructions, this checkpoint, a short activity record, and recent paired tool
  calls/results. Large results are explicitly excerpted. If the retained
  instructions cannot fit, the example stops rather than silently dropping them.

Displayed command output is limited to 48 KiB, but the command continues and up
to 16 MiB is retained in its log. Further output is drained and discarded. Read
logs instead of piping commands to `head` or `tail`. Jobs have a default five-minute
wall timeout, adjustable up to one hour. Escape cancels generation and commands
started in that turn; session exit cancels remaining jobs and their descendants.

For projects requiring an existing dependency installation, expose it explicitly
as read-only and invoke its interpreter by absolute path:

```bash
--sandbox-read-only /path/to/preinstalled/venv
```

Additional directories may be needed for interpreter symlink targets. The flag is
repeatable. Dependency installation, network services, GPU access and host GUI
windows are not supported inside the sandbox. Use headless tests for GUI logic.
For example, ask for `minesweeper.py` with a tkinter GUI and a `--self-test` mode,
then have the model compile it and run its self-tests.

## Containment and resource budgets

Sandboxed file tools, search subprocesses and shell commands use the same isolated
worker boundary. It requires Linux x86-64/AArch64, Landlock ABI 6+, libseccomp and
Bash. Missing support or setup failure never falls back to uncontained execution.
The dedicated example rejects `--unsafe-shell`. The ordinary chat's `--tools`
mode also copies the workspace unless its explicit unsafe opt-out is selected.
Direct library callers must supply a dedicated directory; `shell="none"` is an
explicit file-only mode, not a kernel sandbox.

Landlock restricts file contents to the workspace, the system/Python runtime,
and any explicitly supplied read-only dependency paths. Writes are confined to
the workspace. Seccomp uses a default-deny syscall policy: network sockets, host
display access, metadata mutations, process escape mechanisms and cross-process
control are blocked. The launcher clears the inherited environment and file
descriptors, disables Python site startup, and snapshots its trusted source before
exposing the workspace. File tools use directory descriptors with no symlink
following, bounded regular-file reads and atomic replacement. External hard-link
aliases cause command preflight to refuse the workspace. Original repository
control files are protected by exclusion and by never automatically applying edits.

Default resource controls:

| Resource | Default | Enforcement |
|---|---:|---|
| Job resident memory | 8 GiB | Process-group monitoring, approximately every 250 ms |
| Workspace logical size | 2 GiB | Preflight and approximately one-second monitoring |
| Tasks in a command job | 64 | Group monitoring plus a per-UID kernel process limit |
| Individual file | 256 MiB | Kernel file-size limit |
| Per-process address space | Twice the job memory budget | Kernel limit |
| Log retention | 16 MiB/job, 64 new jobs/session | Parent-side bounds |

Adjust `--sandbox-memory-gib` and `--sandbox-workspace-gib` for larger builds.
Library users can configure `SandboxLimits` directly. Repository copies also have
a total-size budget and reject individual files over 256 MiB; exclude large assets
and expose necessary existing assets read-only instead.

This is practical local containment, not a VM or a multi-tenant execution service.
It shares the host kernel. Monitoring is **best effort**, not a strict cgroup or
filesystem quota: bursts can overshoot, resource scans have overhead, and logical
file size does not account for every filesystem allocation. Filesystem metadata
is not completely hidden. The custom syscall allowlist can reject legitimate
build-tool operations. Code produced by the model is still untrusted when you
choose to run it outside the sandbox. These limits are deliberate; broadening the
syscall policy is not the default response to a failed build.

See the [Linux Landlock documentation](https://docs.kernel.org/userspace-api/landlock.html)
and [seccomp documentation](https://docs.kernel.org/userspace-api/seccomp_filter.html)
for the underlying mechanisms.

## Model settings and implementation

Add `--dflash-path /path/to/assistant` for speculative decoding. Qwen also supports
`--mtp`, `--reasoning-effort medium` and `--no-thinking`. Tokenizer-selected sampling
remains unchanged:

| Model / mode | Temperature | Top-p | Top-k | Presence penalty |
|---|---:|---:|---:|---:|
| Qwen, thinking | 1.0 | 0.95 | 20 | 0.0 |
| Qwen, non-thinking | 0.7 | 0.8 | 20 | 1.5 |
| Muse | 1.0 | 0.95 | 64 | 0.0 |

These follow the [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices)
and [Muse](https://huggingface.co/meta-models/Muse-Glimmer-30B) model cards.
Settings are printed at startup and can be overridden.

The changes live in `runtime/services`: `sandbox.py` handles process containment,
`workspace_files.py` implements isolated file operations, `coding_tools.py` exposes
tools and jobs, `workspace_session.py` manages copies/review, and `agent_context.py`
compacts conversation history. The shared chat loop integrates them. Model kernels,
sampling and speculative decoding do not depend on these services.

## Validation

Regression tests exercise hard-link and directory-swap protection, protected paths,
FIFO rejection, stale hashes and atomic edits, search pagination, process cleanup,
output retention, pipeline failure status, resource budgets, read-only dependencies,
C compilation/execution, session locking/resumption, and context compaction.

Local full-checkpoint smoke tests repaired a failing Python unit test with Qwen
medium thinking and Muse non-thinking. A Qwen non-thinking run with an 8,192-token
context exercises compaction after a large read. These are workflow checks, not a
model reliability benchmark: one Muse thinking attempt hit the repetition guard.
A synthetic 10,000-file repository took about 0.45 seconds to copy, and about
0.1 seconds to fetch its last search/list page on the development machine.
