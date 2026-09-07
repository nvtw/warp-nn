# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Generate and preview Kimodo motions in a resident terminal session.

Models: ``nvidia/Kimodo-SOMA-RP-v1.1``, gated
``meta-llama/Meta-Llama-3-8B-Instruct``,
``McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp``, and its
``-mntp-supervised`` adapter on Hugging Face. Download each with::

    hf download REPOSITORY --local-dir ~/Models/warp-nn/OWNER/NAME

Skinned mode additionally needs ``VAST-AI/SkinTokens`` and a CC0 MakeHuman
male OBJ exported with its joint vertex groups; see ``kimodo_skinned_chat.py``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import shlex
import time
from pathlib import Path

import numpy as np
import warp as wp

try:
    from examples.interactive import (
        TerminalProgress,
        open_output,
        output_path,
        parse_toggle,
    )
except ImportError:
    from interactive import TerminalProgress, open_output, output_path, parse_toggle

from warp_nn.runtime import (
    KimodoConstraints,
    KimodoRunner,
    PanQuadrupedRetargeter,
    decode_motion_features,
    load_makehuman_soma30,
    load_rigged_mesh,
    retarget_soma30_motion,
    save_rigged_mesh,
    save_motion_npz,
    write_motion_html,
)
from warp_nn.runtime.kimodo.constraints import SOMA30_JOINT_NAMES, SOMA30_PARENTS
from warp_nn.runtime.skintokens import SkinTokensPipeline
from warp_nn.utils.paths import application_state_dir


def _parser(*, quadruped=False, skinned=False, description=None):
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("model", type=Path, help="Kimodo checkpoint directory")
    parser.add_argument("text_model", type=Path, help="full Llama-3 8B checkpoint")
    if quadruped:
        parser.add_argument(
            "quadruped_model",
            type=Path,
            help="PAN inference directory containing human, dog, and metadata",
        )
    if skinned:
        parser.add_argument(
            "skin_model", type=Path, help="converted SkinTokens checkpoint directory"
        )
        parser.add_argument(
            "mesh",
            type=Path,
            help="MakeHuman male OBJ containing body and joint vertex groups",
        )
        parser.add_argument(
            "--rig-cache",
            type=Path,
            help="portable rig cache path (default: platform application state)",
        )
        parser.add_argument(
            "--rebuild-rig",
            action="store_true",
            help="ignore and replace an existing rig cache",
        )
        parser.add_argument(
            "--stance-width",
            type=float,
            default=1.0,
            help="lateral hip spacing relative to Kimodo motion (default: 1.0)",
        )
        parser.add_argument(
            "--head-forward",
            type=float,
            default=0.5,
            help="forward neck displacement relative to Kimodo motion (default: 0.5)",
        )
    parser.add_argument(
        "--text-adapter",
        action="append",
        default=[],
        type=Path,
        help="PEFT adapter; repeat for MNTP then supervised adapters",
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="number of seeded motion variations per prompt (default: 1)",
    )
    parser.add_argument("--heading", type=float, default=0.0)
    parser.add_argument("--text-weight", type=float, default=2.0)
    parser.add_argument("--constraint-weight", type=float, default=2.0)
    parser.add_argument(
        "--cfg", choices=("nocfg", "regular", "separated"), default="separated"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cublas", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kimodo-output"),
        help="directory for timestamped HTML previews",
    )
    parser.add_argument(
        "--auto-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="open each finished motion in the default browser (default: on)",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show denoising progress (default: on)",
    )
    return parser


def _help():
    print("Commands:")
    print("  /duration SECONDS   set motion duration (up to 10 seconds)")
    print("  /steps NUMBER       set diffusion steps")
    print("  /seed INTEGER       set the next generation seed")
    print("  /samples NUMBER     set the number of motion variations (1-16)")
    print("  /start [NPZ|last] [SOURCE_FRAME]   constrain the starting pose")
    print("  /end [NPZ|last] [SOURCE_FRAME]     constrain the ending pose")
    print("  /keyframe FRAME [NPZ|last] [SOURCE_FRAME]  add a full-body keyframe")
    print(
        "  /joints FRAME NAMES [NPZ|last] [SOURCE_FRAME] constrain comma-separated joints"
    )
    print("  /effectors FRAME NAMES [NPZ|last] [SOURCE_FRAME] constrain hands/feet")
    print("  /waypoint FRAME X Z [HEADING_DEG]  add a root-path waypoint")
    print("  /path CSV            constrain every frame from x,z[,heading_deg] rows")
    print("  /constraints         list currently constrained frames")
    print("  /clear-constraints   remove all pose and path constraints")
    print("  /save-constraints NPZ  save the current control timeline")
    print("  /load-constraints NPZ  replace it from a saved timeline")
    print("  /open [on|off]      toggle or set automatic browser opening")
    print("  /progress [on|off]  toggle or set progress display")
    print("  /help               show this help")
    print("  /quit (/exit)       close the session")
    print("Prompt sequences: separate consecutive descriptions with ' | '.")


def _try_open(path: Path) -> None:
    try:
        opened = open_output(path)
    except OSError as error:
        print(f"Could not open {path}: {error}")
        return
    if not opened:
        print(f"No desktop opener was found; motion is available at {path}")


def _frames(duration: float, fps: float) -> int:
    frames = round(duration * fps)
    if not 2 <= frames <= 300:
        raise ValueError(f"duration must produce 2-300 frames at {fps:g} FPS")
    return frames


def _pose_source(arguments, last_motion):
    words = shlex.split(arguments)
    source_name = words[0] if words else "last"
    if len(words) > 2:
        raise ValueError("expected [NPZ|last] [SOURCE_FRAME]")
    source = last_motion if source_name.lower() == "last" else Path(source_name)
    if source is None:
        raise ValueError("no previous motion is available; provide a Kimodo NPZ")
    return source, int(words[1]) if len(words) == 2 else -1


def _joint_indices(names):
    by_name = {name.lower(): index for index, name in enumerate(SOMA30_JOINT_NAMES)}
    result = []
    for name in names.split(","):
        key = name.strip().lower()
        if key not in by_name:
            raise ValueError(f"unknown SOMA-30 joint '{name.strip()}'")
        result.append(by_name[key])
    if not result:
        raise ValueError("provide at least one joint name")
    return result


def _rig_cache_path(args):
    if args.rig_cache is not None:
        return args.rig_cache.expanduser()
    digest = hashlib.sha256()
    digest.update(b"makehuman-soma30-meters-v1")
    for path in (args.mesh, args.skin_model / "skintokens.json"):
        digest.update(path.expanduser().read_bytes())
    return (
        application_state_dir()
        / "rigs"
        / f"{args.mesh.stem}-{digest.hexdigest()[:16]}.npz"
    )


def _prepare_rig(args):
    cache = _rig_cache_path(args)
    if cache.is_file() and not args.rebuild_rig:
        print(f"Loading cached SkinTokens rig: {cache}", flush=True)
        return load_rigged_mesh(cache)
    print("Rigging the MakeHuman mesh with SkinTokens (one time)...", flush=True)
    mesh, joints = load_makehuman_soma30(args.mesh)
    pipeline = SkinTokensPipeline(
        args.skin_model,
        dtype=wp.bfloat16,
        device=args.device,
        use_cublas=args.cublas,
    )
    try:
        result = pipeline.rig(
            mesh,
            skeleton_joints=joints,
            skeleton_parents=SOMA30_PARENTS,
            joint_names=SOMA30_JOINT_NAMES,
        )
    finally:
        del pipeline
        gc.collect()
    save_rigged_mesh(cache, result.rig)
    print(f"Cached reusable rig: {cache.resolve()}", flush=True)
    return result.rig


def main(argv=None, *, quadruped=False, skinned=False, description=None):
    if quadruped and skinned:
        raise ValueError("quadruped and skinned modes are mutually exclusive")
    args = _parser(
        quadruped=quadruped, skinned=skinned, description=description
    ).parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    rig = _prepare_rig(args) if skinned else None
    print("Preparing Kimodo and its LLM2Vec text encoder...", flush=True)
    runner = KimodoRunner(
        args.model,
        text_model_path=args.text_model,
        text_adapter_paths=args.text_adapter,
        dtype=wp.bfloat16,
        device=args.device,
        use_cublas=args.cublas,
    )
    retargeter = (
        PanQuadrupedRetargeter(args.quadruped_model, device=args.device)
        if quadruped
        else None
    )
    duration = args.duration
    frame_count = _frames(duration, runner.config.fps)
    constraints = KimodoConstraints.empty(frame_count, runner.config.joints)
    last_motion = None
    steps = args.steps
    next_seed = args.seed
    num_samples = args.num_samples
    if not 1 <= num_samples <= 16:
        raise ValueError("--num-samples must be between 1 and 16")
    auto_open = args.auto_open
    show_progress = args.progress
    subject = (
        "quadruped motions"
        if quadruped
        else ("skinned human motions" if skinned else "motions")
    )
    print(f"Ready on {runner.device}; weights stay loaded between prompts.")
    if retargeter is not None:
        print("PAN dog retargeting is enabled; its frame plans remain cached.")
    print(
        f"Generated {subject} are written to {args.output_dir.expanduser().resolve()}."
    )
    _help()

    while True:
        try:
            prompt = input("Motion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        command, _, value = prompt.partition(" ")
        command = command.lower()
        if command in ("/quit", "/exit") and not value:
            break
        if command == "/help" and not value:
            _help()
            continue
        if command in ("/open", "/progress"):
            current = auto_open if command == "/open" else show_progress
            try:
                enabled = parse_toggle(value, current)
            except ValueError as error:
                print(f"Usage: {command} [on|off] ({error})")
                continue
            if command == "/open":
                auto_open = enabled
                label = "Automatic opening"
            else:
                show_progress = enabled
                label = "Progress"
            print(f"{label} is {'on' if enabled else 'off'}.")
            continue
        if command == "/duration":
            try:
                candidate = float(value)
                _frames(candidate, runner.config.fps)
            except ValueError:
                print("Usage: /duration SECONDS (approximately 0.07-10)")
                continue
            duration = candidate
            frame_count = _frames(duration, runner.config.fps)
            if constraints.mask.any():
                print(
                    "Duration changed, so existing frame-based constraints were cleared."
                )
            constraints = KimodoConstraints.empty(frame_count, runner.config.joints)
            print(f"Duration is {duration:g}s.")
            continue
        if command == "/steps":
            try:
                candidate = int(value)
                if not 1 <= candidate <= runner.config.diffusion_steps:
                    raise ValueError
            except ValueError:
                print(f"Usage: /steps NUMBER (1-{runner.config.diffusion_steps})")
                continue
            steps = candidate
            print(f"Diffusion steps: {steps}.")
            continue
        if command == "/seed":
            try:
                next_seed = int(value)
            except ValueError:
                print("Usage: /seed INTEGER")
                continue
            print(f"Next seed is {next_seed}.")
            continue
        if command == "/samples":
            try:
                candidate = int(value)
                if not 1 <= candidate <= 16:
                    raise ValueError
            except ValueError:
                print("Usage: /samples NUMBER (1-16)")
                continue
            num_samples = candidate
            print(f"Motion variations per prompt: {num_samples}.")
            continue
        if command == "/clear-constraints" and not value:
            constraints.clear()
            print("All constraints cleared.")
            continue
        if command in ("/save-constraints", "/load-constraints"):
            try:
                words = shlex.split(value)
                if len(words) != 1:
                    raise ValueError("expected one NPZ path")
                path = Path(words[0]).expanduser()
                if command == "/save-constraints":
                    path.parent.mkdir(parents=True, exist_ok=True)
                    constraints.save(path)
                    action = "Saved"
                else:
                    loaded = KimodoConstraints.load(path)
                    if (
                        loaded.frames != frame_count
                        or loaded.joints != runner.config.joints
                    ):
                        raise ValueError(
                            f"archive is {loaded.frames} frames/{loaded.joints} joints; "
                            f"this session is {frame_count}/{runner.config.joints}"
                        )
                    constraints = loaded
                    action = "Loaded"
            except (OSError, ValueError) as error:
                print(f"Usage: {command} NPZ ({error})")
                continue
            print(f"{action} constraints: {path.resolve()}")
            continue
        if command == "/constraints" and not value:
            constrained = constraints.constrained_frames.tolist()
            print(
                f"Constrained frames: {constrained}"
                if constrained
                else "No active constraints."
            )
            continue
        if command in ("/start", "/end"):
            try:
                source, source_frame = _pose_source(value, last_motion)
                if command == "/start":
                    constraints.start_pose(source, source_frame=source_frame)
                    target = 0
                else:
                    constraints.end_pose(source, source_frame=source_frame)
                    target = frame_count - 1
            except (OSError, ValueError) as error:
                print(f"Usage: {command} [NPZ|last] [SOURCE_FRAME] ({error})")
                continue
            print(f"Full-body pose constrained at frame {target}.")
            continue
        if command in ("/keyframe", "/joints", "/effectors"):
            try:
                words = shlex.split(value)
                minimum = 2 if command in ("/joints", "/effectors") else 1
                if len(words) < minimum or len(words) > minimum + 2:
                    raise ValueError("invalid number of arguments")
                target = int(words[0])
                offset = 2 if command in ("/joints", "/effectors") else 1
                source_words = words[offset:]
                source, source_frame = _pose_source(
                    " ".join(shlex.quote(word) for word in source_words), last_motion
                )
                joints = _joint_indices(words[1]) if command == "/joints" else None
                if command == "/effectors":
                    constraints.end_effectors(
                        target,
                        source,
                        [name.strip() for name in words[1].split(",")],
                        source_frame=source_frame,
                    )
                else:
                    constraints.pose(
                        target,
                        source,
                        source_frame=source_frame,
                        joints=joints,
                        rotations=joints is not None,
                    )
            except (OSError, ValueError) as error:
                usage = (
                    f"{command} FRAME NAME[,NAME...] [NPZ|last] [SOURCE_FRAME]"
                    if command in ("/joints", "/effectors")
                    else "/keyframe FRAME [NPZ|last] [SOURCE_FRAME]"
                )
                print(f"Usage: {usage} ({error})")
                continue
            label = (
                "End effectors"
                if command == "/effectors"
                else ("Selected joints" if joints is not None else "Full body")
            )
            print(f"{label} constrained at frame {target}.")
            continue
        if command == "/waypoint":
            try:
                words = shlex.split(value)
                if len(words) not in (3, 4):
                    raise ValueError("expected FRAME X Z [HEADING_DEG]")
                target, x, z = int(words[0]), float(words[1]), float(words[2])
                heading = np.deg2rad(float(words[3])) if len(words) == 4 else None
                constraints.root_path(target, [[x, z]], heading=heading)
            except ValueError as error:
                print(f"Usage: /waypoint FRAME X Z [HEADING_DEG] ({error})")
                continue
            print(f"Root waypoint constrained at frame {target}.")
            continue
        if command == "/path":
            try:
                words = shlex.split(value)
                if len(words) != 1:
                    raise ValueError("expected one CSV path")
                points = np.loadtxt(words[0], delimiter=",", ndmin=2, dtype=np.float32)
                if points.shape not in ((frame_count, 2), (frame_count, 3)):
                    raise ValueError(
                        f"CSV must have {frame_count} rows and 2 or 3 columns"
                    )
                heading = np.deg2rad(points[:, 2]) if points.shape[1] == 3 else None
                constraints.root_path(
                    np.arange(frame_count), points[:, :2], heading=heading
                )
            except (OSError, ValueError) as error:
                print(f"Usage: /path CSV ({error})")
                continue
            print(f"Dense {frame_count}-frame root path constrained.")
            continue
        if prompt.startswith("/"):
            print(f"Unknown command: {command}. Use /help for available commands.")
            continue

        segments = [segment.strip() for segment in prompt.split(" | ")]
        if any(not segment for segment in segments):
            print("Prompt sequence segments around ' | ' must not be empty.")
            continue
        frames = frame_count
        active_constraints = constraints if constraints.mask.any() else None
        if len(segments) > 1 and active_constraints is not None:
            sequence_constraints = KimodoConstraints.empty(
                frame_count * len(segments), runner.config.joints
            )
            sequence_constraints.overlay(active_constraints)
            active_constraints = sequence_constraints
        progress = TerminalProgress("Motion", enabled=show_progress)
        print(
            f"Generating {duration:g}s ({frames} frames, {steps} steps, "
            f"{num_samples} sample{'s' if num_samples != 1 else ''}, "
            f"seed {next_seed})...",
            flush=True,
        )
        started = time.perf_counter()
        try:
            generation_options = dict(
                denoising_steps=steps,
                cfg_type=args.cfg,
                text_weight=args.text_weight,
                constraint_weight=args.constraint_weight,
                heading=np.array([args.heading], dtype=np.float32),
                constraints=active_constraints,
                seed=next_seed,
                progress=progress,
            )
            features = (
                runner.generate(
                    segments[0], frames, num_samples=num_samples, **generation_options
                )
                if len(segments) == 1
                else np.concatenate(
                    [
                        np.asarray(
                            runner.generate_sequence(
                                segments,
                                [frames] * len(segments),
                                **(generation_options | {"seed": next_seed + sample}),
                            )
                        )
                        for sample in range(num_samples)
                    ]
                )
            )
            wp.synchronize_device(runner.device)
            generation_seconds = time.perf_counter() - started
            decoded = decode_motion_features(
                features,
                runner.stats,
                runner.config.joints,
                constraints=active_constraints,
            )
            batched = decoded["posed_joints"].ndim == 4
            sample_motions = [
                {
                    name: value[sample]
                    if batched
                    and isinstance(value, np.ndarray)
                    and value.shape[0] == num_samples
                    else value
                    for name, value in decoded.items()
                }
                for sample in range(num_samples)
            ]
            last_motion = sample_motions[0]
            retarget_started = time.perf_counter()
            viewed_motions = (
                [retargeter.retarget(motion) for motion in sample_motions]
                if retargeter is not None
                else (
                    [
                        retarget_soma30_motion(
                            motion,
                            rig.rest_joints,
                            stance_width=args.stance_width,
                            head_forward=args.head_forward,
                        )
                        for motion in sample_motions
                    ]
                    if rig is not None
                    else sample_motions
                )
            )
            retarget_seconds = (
                time.perf_counter() - retarget_started
                if retargeter is not None or rig is not None
                else 0.0
            )
            base = output_path(args.output_dir, prompt, ".html").resolve()
            destinations = []
            for sample, (source_motion, viewed_motion) in enumerate(
                zip(sample_motions, viewed_motions)
            ):
                destination = (
                    base
                    if num_samples == 1
                    else base.with_name(f"{base.stem}-{sample + 1:02d}{base.suffix}")
                )
                save_motion_npz(
                    destination.with_suffix(".npz"),
                    source_motion,
                    fps=runner.config.fps,
                )
                write_motion_html(
                    destination,
                    viewed_motion,
                    fps=runner.config.fps,
                    prompt=prompt,
                    seed=next_seed + sample,
                    generation_seconds=generation_seconds + retarget_seconds,
                    mesh=rig,
                    label=(
                        "Kimodo · Quadruped"
                        if quadruped
                        else ("Kimodo · SkinTokens" if skinned else "Kimodo")
                    ),
                )
                destinations.append(destination)
        except Exception as error:
            print(f"Generation failed: {error}")
            continue
        timing = f"{generation_seconds:.2f}s"
        if retargeter is not None or rig is not None:
            timing += f" + {retarget_seconds:.2f}s retargeting"
        print(f"Kimodo> {', '.join(map(str, destinations))} ({timing})")
        next_seed += num_samples
        if auto_open:
            for destination in destinations:
                _try_open(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
