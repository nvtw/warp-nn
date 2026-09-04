# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-style multimodal prompt preparation for Nemotron Omni."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..formats.image import load_rgb_image
from ..formats.media import decode_audio_mono
from .audio import (
    NemotronAudioConfig,
    ParakeetFeatures,
    parakeet_subsampled_length,
    preprocess_parakeet_audio,
    preprocess_parakeet_wav,
)
from .vision import NemotronImage, preprocess_nemotron_image
from .video import NemotronVideo, preprocess_nemotron_video, video_prompt_chunks


@dataclass(frozen=True)
class NemotronMultimodalPrompt:
    """Tokenized chat plus preprocessed media and its placeholder locations."""

    token_ids: tuple[int, ...]
    images: tuple[NemotronImage, ...]
    image_starts: tuple[int, ...]
    audios: tuple[ParakeetFeatures, ...] = ()
    audio_starts: tuple[int, ...] = ()
    videos: tuple[NemotronVideo, ...] = ()
    video_starts: tuple[tuple[int, ...], ...] = ()

    @property
    def media(self):
        """All media objects, for shared interactive cache invalidation."""
        return self.images + self.audios + self.videos


class NemotronMultimodalProcessor:
    """Prepare Nemotron image/audio/video chat prompts with optional PyAV."""

    def __init__(
        self,
        tokenizer,
        audio_config: NemotronAudioConfig | None = None,
        *,
        video_temporal_patch_size: int = 2,
        video_target_patches: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.audio_config = audio_config
        self.video_temporal_patch_size = int(video_temporal_patch_size)
        self.video_target_patches = int(video_target_patches)
        image = tokenizer.encode("<image>")
        if len(image) != 1:
            raise ValueError("Nemotron tokenizer does not define one <image> token")
        self.image_token_id = int(image[0])
        sound = tokenizer.encode("<so_embedding>")
        if len(sound) != 1:
            raise ValueError(
                "Nemotron tokenizer does not define one <so_embedding> token"
            )
        self.audio_token_id = int(sound[0])

    def encode_chat(
        self, messages: Sequence[Mapping[str, object]], **kwargs
    ) -> NemotronMultimodalPrompt:
        images = []
        audios = []
        videos = []
        visual_records = []
        transformed = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, Sequence) or isinstance(content, str):
                transformed.append(dict(message))
                continue
            pieces = []
            for item in content:
                if not isinstance(item, Mapping):
                    raise ValueError("multimodal content entries must be objects")
                kind = item.get("type")
                if kind == "text":
                    pieces.append(str(item.get("text", "")))
                    continue
                if kind in ("audio", "input_audio"):
                    source = item.get("audio", item.get("input_audio"))
                    if isinstance(source, Mapping):
                        source = source.get("path", source.get("url"))
                    if isinstance(source, (str, Path)):
                        media = (
                            preprocess_parakeet_wav(source)
                            if Path(source).suffix.lower() == ".wav"
                            else preprocess_parakeet_audio(decode_audio_mono(source))
                        )
                    else:
                        media = preprocess_parakeet_audio(source)
                    config = self.audio_config
                    if config is None:
                        raise ValueError(
                            "audio_config is required to prepare Nemotron audio"
                        )
                    tokens = parakeet_subsampled_length(
                        int(media.attention_mask.sum()), config
                    )
                    audios.append(media)
                    pieces.append("<so_start>" + "<so_embedding>" * tokens + "<so_end>")
                    continue
                if kind in ("video", "video_url"):
                    source = item.get("video", item.get("video_url"))
                    if isinstance(source, Mapping):
                        source = source.get("path", source.get("url"))
                    media = preprocess_nemotron_video(
                        source,
                        fps=float(item.get("fps", 1.0)),
                        max_frames=int(item.get("max_frames", 128)),
                        temporal_patch_size=self.video_temporal_patch_size,
                        target_patches=self.video_target_patches,
                    )
                    videos.append(media)
                    visual_records.append(
                        (
                            "video",
                            len(videos) - 1,
                            (media.tokens_per_group,) * media.groups,
                        )
                    )
                    pieces.append("\n".join(video_prompt_chunks(media)))
                    continue
                if kind not in ("image", "image_url"):
                    raise ValueError(f"unsupported Nemotron media type '{kind}'")
                source = item.get("image", item.get("image_url"))
                if isinstance(source, Mapping):
                    source = source.get("url")
                if isinstance(source, (str, Path)):
                    source = load_rgb_image(source)
                media = preprocess_nemotron_image(source)
                images.append(media)
                visual_records.append(("image", len(images) - 1, (media.tokens,)))
                pieces.append("<img>" + "<image>" * media.tokens + "</img>")
            copied = dict(message)
            copied["content"] = "".join(pieces)
            transformed.append(copied)

        token_ids = tuple(
            self.tokenizer.encode(self.tokenizer.format_chat(transformed, **kwargs))
        )
        starts = [None] * len(images)
        video_starts = [None] * len(videos)
        cursor = 0
        for kind, index, blocks in visual_records:
            found = []
            for tokens in blocks:
                while (
                    cursor < len(token_ids) and token_ids[cursor] != self.image_token_id
                ):
                    cursor += 1
                if (
                    token_ids[cursor : cursor + tokens]
                    != (self.image_token_id,) * tokens
                ):
                    raise ValueError(
                        "formatted prompt has fewer visual tokens than features"
                    )
                found.append(cursor)
                cursor += tokens
            if kind == "image":
                starts[index] = found[0]
            else:
                video_starts[index] = tuple(found)
        audio_starts = []
        cursor = 0
        for media in audios:
            tokens = parakeet_subsampled_length(
                int(media.attention_mask.sum()), self.audio_config
            )
            while cursor < len(token_ids) and token_ids[cursor] != self.audio_token_id:
                cursor += 1
            if token_ids[cursor : cursor + tokens] != (self.audio_token_id,) * tokens:
                raise ValueError(
                    "formatted prompt has fewer audio tokens than features"
                )
            audio_starts.append(cursor)
            cursor += tokens
        return NemotronMultimodalPrompt(
            token_ids,
            tuple(images),
            tuple(starts),
            tuple(audios),
            tuple(audio_starts),
            tuple(videos),
            tuple(video_starts),
        )
