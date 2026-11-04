"""Validated JSON configuration launcher for the experimental llm-server."""
import argparse
import json
import os
from pathlib import Path

VALUE_OPTIONS = {
    "host", "port", "server_model_name", "max_tokens", "max_output_tokens",
    "temperature", "concurrent_requests", "max_queue_depth", "prefix_cache_mib",
    "prefill_chunk_size", "request_timeout_seconds", "api_key_env", "prefill_chunks_per_step",
    "stateful_prefill_model",
}
FLAGS = {"functional_batch", "no_thinking", "verbose", "disable_persistent_prefill"}


def command(config_path, binary=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(config) - VALUE_OPTIONS - FLAGS - {"model"}
    if unknown:
        raise ValueError("Unknown configuration fields: " + ", ".join(sorted(unknown)))
    if not isinstance(config.get("model"), str):
        raise ValueError("model must be an asset path")
    model = Path(config["model"]).expanduser()
    if not model.is_absolute():
        model = config_path.parent / model
    if not (model / "metadata.json").is_file():
        raise ValueError("Model directory is missing metadata.json")
    if config.get("stateful_prefill_model") is not None:
        state_path = config["stateful_prefill_model"]
        if not isinstance(state_path, str) or not config.get("functional_batch"):
            raise ValueError("stateful_prefill_model requires a path and functional_batch")
        state_path = Path(state_path).expanduser()
        if not state_path.is_absolute():
            state_path = config_path.parent / state_path
        if not (state_path / "batch-manifest.json").is_file():
            raise ValueError("State bundle is missing batch-manifest.json")
        config["stateful_prefill_model"] = str(state_path.resolve())
    root = Path(__file__).resolve().parent.parent
    executable = Path(binary) if binary else root / ".build/out/Products/Release/llm-server"
    if not executable.is_file():
        raise ValueError("Build llm-server in release mode first")
    args = [str(executable), "--model", str(model.resolve())]
    for key in sorted(VALUE_OPTIONS):
        value = config.get(key)
        if value is not None:
            if isinstance(value, (dict, list, bool)):
                raise ValueError(key + " requires a scalar value")
            args.extend(["--" + key.replace("_", "-"), str(value)])
    for key in sorted(FLAGS):
        value = config.get(key, False)
        if not isinstance(value, bool):
            raise ValueError(key + " must be a boolean")
        if value:
            args.append("--" + key.replace("_", "-"))
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    argv = command(args.config, args.binary)
    if args.check:
        print(json.dumps(argv))
    else:
        os.execvpe(argv[0], argv, os.environ.copy())


if __name__ == "__main__":
    main()
