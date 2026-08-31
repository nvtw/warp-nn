# Model atlas

Create an interactive map of a local GGUF or safetensors checkpoint:

```bash
.venv/bin/python -m warp_nn.runtime.analysis.model_graph /path/to/model
```

The command reads checkpoint headers only, writes a standalone HTML report to a
new temporary directory, and opens it in the default browser. It does not load
weights into CPU arrays or GPU memory. Preserve a report at a known location, or
generate it without a browser, with:

```bash
.venv/bin/python -m warp_nn.runtime.analysis.model_graph /path/to/model --output model-map.html --no-open
```

The overview follows the model's computation from embeddings through its layers
to the output. Click a component for an explanation and exact storage statistics.
Use search (`/`) to jump to a layer or exact weight name. Learned tensors remain
available from each component's sidebar without crowding the architecture map.
