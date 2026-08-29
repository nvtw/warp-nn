# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Fixed-buffer LoRA SwiGLU feed-forward training composition."""

from functools import lru_cache

import warp as wp

from .adapters import LoRAAdapterCollection


@lru_cache(maxsize=None)
def _swiglu_kernels(dtype: type):
    DTYPE = dtype

    @wp.func
    def sigmoid(value: wp.float32):
        if value >= wp.float32(0.0):
            return wp.float32(1.0) / (wp.float32(1.0) + wp.exp(-value))
        exponential = wp.exp(value)
        return exponential / (wp.float32(1.0) + exponential)

    @wp.kernel(enable_backward=False, module="unique")
    def forward(
        gate: wp.array1d(dtype=DTYPE),
        up: wp.array1d(dtype=DTYPE),
        output: wp.array1d(dtype=DTYPE),
    ):
        index = wp.tid()
        gate_value = wp.float32(gate[index])
        output[index] = DTYPE(gate_value * sigmoid(gate_value) * wp.float32(up[index]))

    @wp.kernel(enable_backward=False, module="unique")
    def backward(
        gate: wp.array1d(dtype=DTYPE),
        up: wp.array1d(dtype=DTYPE),
        output_grad: wp.array1d(dtype=DTYPE),
        gate_grad: wp.array1d(dtype=DTYPE),
        up_grad: wp.array1d(dtype=DTYPE),
    ):
        index = wp.tid()
        gate_value = wp.float32(gate[index])
        up_value = wp.float32(up[index])
        gradient = wp.float32(output_grad[index])
        probability = sigmoid(gate_value)
        gate_grad[index] = DTYPE(
            gradient
            * up_value
            * probability
            * (wp.float32(1.0) + gate_value * (wp.float32(1.0) - probability))
        )
        up_grad[index] = DTYPE(gradient * gate_value * probability)

    @wp.kernel(enable_backward=False, module="unique")
    def sum_inputs(
        gate: wp.array1d(dtype=DTYPE),
        up: wp.array1d(dtype=DTYPE),
        output: wp.array1d(dtype=wp.float32),
    ):
        index = wp.tid()
        output[index] = wp.float32(gate[index]) + wp.float32(up[index])

    for kernel in (forward, backward, sum_inputs):
        kernel.module.options["enable_backward"] = False
    return forward, backward, sum_inputs


class LoRASwiGLUPlan:
    """Compose named gate/up/down LoRA projections with explicit SwiGLU backward."""

    def __init__(
        self,
        adapters: LoRAAdapterCollection,
        *,
        gate: str,
        up: str,
        down: str,
    ):
        names = (gate, up, down)
        if len(set(names)) != 3 or any(name not in adapters.targets for name in names):
            raise ValueError("SwiGLU projection names must be distinct adapter targets")
        targets = tuple(adapters.targets[name] for name in names)
        dtype = targets[0].weight.dtype
        device = targets[0].weight.device
        rows = targets[0].plan.rows
        if any(
            target.weight.dtype != dtype
            or target.weight.device != device
            or target.plan.rows != rows
            for target in targets
        ):
            raise ValueError("SwiGLU targets must share dtype, device, and rows")
        intermediate, hidden = targets[0].weight.shape
        expected = (
            (intermediate, hidden),
            (intermediate, hidden),
            (hidden, intermediate),
        )
        actual = tuple(target.weight.shape for target in targets)
        if actual != expected:
            raise ValueError(
                f"SwiGLU projection shapes must be {expected}, got {actual}"
            )

        self.adapters = adapters
        self.names = names
        self.dtype = dtype
        self.device = device
        self.rows = rows
        self.hidden = hidden
        self.intermediate = intermediate
        self.activated = wp.empty((rows, intermediate), dtype=dtype, device=device)
        self.gate_grad = wp.empty_like(self.activated)
        self.up_grad = wp.empty_like(self.activated)
        self.input_grad = wp.empty((rows, hidden), dtype=wp.float32, device=device)

    @property
    def output(self) -> wp.array:
        """Return the fixed down-projection output buffer."""
        return self.adapters.targets[self.names[2]].plan.output

    def forward(self, x: wp.array) -> wp.array:
        """Execute gate/up projections, SwiGLU, and the down projection."""
        gate_name, up_name, down_name = self.names
        gate = self.adapters.forward(gate_name, x)
        up = self.adapters.forward(up_name, x)
        wp.launch(
            _swiglu_kernels(self.dtype)[0],
            dim=self.activated.size,
            inputs=[gate.flatten(), up.flatten()],
            outputs=[self.activated.flatten()],
            device=self.device,
        )
        return self.adapters.forward(down_name, self.activated)

    def backward(
        self, x: wp.array, grad_output: wp.array, *, accumulate: bool = False
    ) -> wp.array:
        """Run down/SwiGLU/gate/up backward and return fixed FP32 dX."""
        gate_name, up_name, down_name = self.names
        activated_grad = self.adapters.backward(
            down_name, self.activated, grad_output, accumulate=accumulate
        )
        gate = self.adapters.targets[gate_name].plan.output
        up = self.adapters.targets[up_name].plan.output
        wp.launch(
            _swiglu_kernels(self.dtype)[1],
            dim=self.activated.size,
            inputs=[gate.flatten(), up.flatten(), activated_grad.flatten()],
            outputs=[self.gate_grad.flatten(), self.up_grad.flatten()],
            device=self.device,
        )
        gate_input = self.adapters.backward(
            gate_name, x, self.gate_grad, accumulate=accumulate
        )
        up_input = self.adapters.backward(
            up_name, x, self.up_grad, accumulate=accumulate
        )
        wp.launch(
            _swiglu_kernels(self.dtype)[2],
            dim=self.input_grad.size,
            inputs=[gate_input.flatten(), up_input.flatten()],
            outputs=[self.input_grad.flatten()],
            device=self.device,
        )
        return self.input_grad
