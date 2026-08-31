# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free interactive client for the warp-nn OpenAI endpoint."""

import argparse
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        value = "http://" + value
    return value if value.endswith("/v1") else value + "/v1"


def _json_request(url: str, api_key: str | None, body=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    with urlopen(Request(url, data=data, headers=headers), timeout=3600) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", help="Server base URL, such as http://192.168.1.5:8000/v1"
    )
    parser.add_argument("--model", help="Model ID; discovered automatically if omitted")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("WARP_NN_API_KEY"),
        help="Bearer token; defaults to WARP_NN_API_KEY",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    url = _base_url(
        args.url or input("Server URL (for example http://192.168.1.5:8000/v1): ")
    )
    try:
        models = _json_request(url + "/models", args.api_key)
        model = args.model or models["data"][0]["id"]
        print(f"Connected to {url} · model {model} · /quit to exit")
        messages = []
        while True:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            if prompt in ("/quit", "/exit"):
                break
            messages.append({"role": "user", "content": prompt})
            response = _json_request(
                url + "/chat/completions",
                args.api_key,
                {
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": args.max_tokens,
                },
            )
            message = response["choices"][0]["message"]
            reasoning = message.get("reasoning_content")
            if reasoning:
                print(f"Reasoning: {reasoning}")
            content = message.get("content") or ""
            print(f"Assistant: {content}")
            messages.append(message)
    except (HTTPError, URLError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"Connection or API error: {error}") from error


if __name__ == "__main__":
    main()
