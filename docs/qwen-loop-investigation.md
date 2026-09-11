# Qwen GGUF repetition investigation

## Root cause

The Qwen 3.5/3.8 GGUF runner incorrectly permuted query and key channels from an
assumed interleaved layout to a split-half layout before Q/K normalization and
RoPE. The official GGUF conversion preserves these full-attention channel
orders. Interleaved **MRoPE frequency selection** and the GGUF **DeltaNet value-head
reordering** are separate operations; neither justifies permuting Q/K channels.

The incorrect permutation affected prefill, ordinary decode, speculative
verification, and independent batched decode. It corrupted full attention while
the recurrent layers still produced superficially plausible text. Muse uses a
different model path and was unaffected.

The fix removes the permutation from both Qwen attention execution paths. A
regression test converts a tiny safetensors checkpoint using the GGUF converter's
norm, decay, and convolution transforms, then checks matching prompt/decode logits
across formats. It failed before the fix and passes afterward. Reference conversion:
[llama.cpp converter](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py),
`_LinearAttentionVReorderBase` and `Qwen3NextModel`.

## Full checkpoint reproduction

Tested on an RTX PRO 6000 Blackwell with the local BF16 Qwen3.8-27B checkpoint,
seed 42, a 16,384-token cache, 256-token prefill chunks, and a 10,000-token response
limit. Prompt:

> Write a full minesweeper game in Python including a GUI. Include all code in your answer.

The official sampling settings were unchanged (see [coding-agent.md](coding-agent.md)).
Thinking effort was medium. DFlash used the published Qwen3.8-27B-DFlash2 assistant.

| Mode | Before fix | After fix |
|---|---|---|
| Medium thinking, ordinary | Repetition stop at 6,214 tokens | EOS at 8,978 tokens |
| Medium thinking, DFlash | Repetition stop at 9,598 tokens | EOS at 5,759 tokens |
| Non-thinking, ordinary | EOS at 4,716 tokens | EOS at 3,679 tokens |
| Non-thinking, DFlash | Repetition stop at 3,004 tokens | EOS at 2,993 tokens |

All four corrected outputs passed Python compilation inside the sandbox. These
are generation/syntax checks, not a claim that every generated GUI is bug-free.
Sampled ordinary and speculative outputs can differ because of small numerical
differences in token versus multi-row target evaluation.

Qwen also completed both modes with embedded MTP (5,312 thinking tokens and
3,506 non-thinking tokens).

Muse completed the same prompt with its official defaults, both ordinarily
(1,819 tokens) and with its DFlash assistant (2,307 tokens).

## Additional corrections

The sampling audit also found that bounded top-k reads selected the first row
from a multi-row logits tensor, and greedy sampling skipped presence penalties.
Both are fixed with focused regressions. These were separate bugs, not the cause
of the initial medium-thinking reproduction. GPU top-k supports Muse's top-64
policy; sampling validation now rejects nonfinite or invalid parameters before
fast paths.

Qwen tool templates now retain low/xhigh effort instructions alongside an explicit
system message and preserve reasoning within the current multi-step tool turn.
The inference fix does not rely on changing sampling defaults, adding penalties,
or forcing a thinking-budget cutoff.

## Coding workflow validation

Both models used the actual tools to run a seeded `NameError` in `broken.py`,
edit the file, and verify that it printed `42`. Both wrote a tkinter Minesweeper
program through `write_file`. Muse's generated program passed its syntax check
and six headless tests. Qwen's generated self-test contained bugs; subsequent
model tool calls repaired them and both checks passed. The GUI windows themselves
were not exercised: the sandbox intentionally has no host display connection.

The final focused suite passed 122 tests covering model inference, format parity,
sampling, chat history, tool parsing, sandbox escapes/lifecycle, HTTP serving,
and batching. Three additional GPU top-k operator tests also passed. Ruff and
`git diff --check` passed. Tool parsing now preserves indentation/trailing newlines
and schema-declared string values, including text that looks like JSON literals.
