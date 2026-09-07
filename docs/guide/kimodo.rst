Kimodo text-to-motion inference
===============================

``warp-nn`` can execute the released Kimodo two-stage diffusion architecture
without PyTorch, Transformers, PEFT, or a YAML package. Dense projections use
the normal runtime planner, bidirectional encoder attention is shared with the
LLM2Vec conditioner, and fixed-shape denoising is CUDA-graph captured.

Required local files
--------------------

No downloader is built into the runner. Prepare these directories yourself:

* A Kimodo checkpoint directory containing ``config.yaml``,
  ``model.safetensors``, and ``stats/motion/{global_root,local_root,body}``.
* The full ``meta-llama/Meta-Llama-3-8B-Instruct`` base checkpoint, including
  config, tokenizer, and safetensors shards.
* The ``McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`` adapter directory.
* The ``McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised`` adapter
  directory.

The two LLM2Vec repositories are PEFT adapters, not substitutes for the full
Llama base checkpoint. They must be supplied in MNTP-then-supervised order.

.. code-block:: bash

    python examples/kimodo.py \
      /models/Kimodo-SOMA-RP-v1.1 \
      /models/Meta-Llama-3-8B-Instruct \
      "a person walks forward and waves" \
      --text-adapter /models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp \
      --text-adapter /models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised \
      --frames 150 --steps 100 --output motion.npz

The output NPZ contains model-native features, root positions, posed joint
positions, global rotation matrices, velocities, headings, and foot contacts.
The portable decoder intentionally does not require Kimodo's renderer or
skeleton classes.

Interactive browser preview
---------------------------

``kimodo_chat.py`` keeps Kimodo and its LLM2Vec conditioner loaded. Each normal
line entered at ``Motion>`` generates a new single-file HTML preview and opens
it in the default browser. The exact joint positions and foot contacts are
embedded in the page; no NPZ or server is needed. The viewer loads its pinned
Three.js renderer from the web when opened.

.. code-block:: bash

    python examples/kimodo_chat.py \
      /models/Kimodo-SOMA-RP-v1.1 \
      /models/Meta-Llama-3-8B-Instruct \
      --text-adapter /models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp \
      --text-adapter /models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised

Use ``/duration``, ``/steps``, ``/seed``, ``/open`` and ``/progress`` to adjust
the resident session. The browser preview provides playback, scrubbing,
looping, speed control, an orbit camera, root following and camera reset.
