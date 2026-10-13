#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import yaml
from openai import OpenAI


def load_model_cfg(config_path: Path, model_name: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    models = cfg.get("models", {})
    if model_name not in models:
        raise ValueError(
            f"模型 '{model_name}' 不存在于 {config_path} 的 models 中。可用模型: {list(models.keys())}"
        )
    m = models[model_name]
    base_url = (m.get("base_url") or "").rstrip("/")
    if base_url.endswith("/api"):
        base_url = base_url[:-4] + "/v1"
    elif base_url and "/v1" not in base_url:
        base_url = f"{base_url}/v1"
    return {
        "model": m.get("model"),
        "base_url": base_url or "https://api.openai.com/v1",
        "api_key": m.get("api_key", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="测试 gpt4 模型连接是否成功")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config2.yaml"),
        help="配置文件路径，默认 config/config2.yaml",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt4",
        help="models 中的模型键名，默认 gpt4",
    )
    args = parser.parse_args()

    try:
        model_cfg = load_model_cfg(args.config, args.model_name)
        client = OpenAI(
            base_url=model_cfg["base_url"],
            api_key=model_cfg["api_key"],
        )
        resp = client.chat.completions.create(
            model=model_cfg["model"],
            temperature=0,
            max_tokens=16,
            messages=[
                {"role": "system", "content": "You are a health-check assistant."},
                {"role": "user", "content": "Reply with exactly: OK"},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        print("[SUCCESS] 连接成功")
        print(f"model_name={args.model_name}")
        print(f"model={model_cfg['model']}")
        print(f"base_url={model_cfg['base_url']}")
        print(f"response={content}")
        return 0
    except Exception as e:
        print(f"[FAILED] 连接失败: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
