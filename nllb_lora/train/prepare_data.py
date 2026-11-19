"""Verify frozen bilingual inputs and stage a minimal Colab directory bundle."""

import argparse
from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path
import re
import shutil
import importlib.metadata
import math
import platform
import time


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


LANGUAGE_PAIRS = (("zho_Hans", "vie_Latn"), ("vie_Latn", "zho_Hans"))


def direction_specs(config):
    """Translation directions of an experiment.

    Legacy single-direction configs yield one entry named None. A shared adapter
    lists ``model.directions``; the first must equal the primary
    source/target pair so its row IDs and file roles match the single-direction format.
    """
    spec = config["model"]
    primary = (spec["source_language"], spec["target_language"])
    declared = spec.get("directions")
    if not declared:
        return [{"name": None, "source_language": primary[0], "target_language": primary[1], "swap": False}]
    result, names, pairs = [], set(), set()
    for item in declared:
        pair = (item["source_language"], item["target_language"])
        name = item["name"]
        if pair not in LANGUAGE_PAIRS or pair in pairs or not isinstance(name, str) or not re.fullmatch(r"[a-z]+-[a-z]+", name) or name in names:
            raise ValueError(f"Invalid or duplicate direction: {item}")
        pairs.add(pair)
        names.add(name)
        result.append({"name": name, "source_language": pair[0], "target_language": pair[1], "swap": pair != primary})
    if (result[0]["source_language"], result[0]["target_language"]) != primary:
        raise ValueError("The first direction must equal the primary model source/target pair")
    return result


def is_shared(config):
    return len(direction_specs(config)) > 1


def direction_row_id(split, number, position, direction):
    """The first direction keeps the single-direction ``split:N`` ID; later ones add ``@name``."""
    return f"{split}:{number}" if position == 0 else f"{split}:{number}@{direction['name']}"


def train_example_count(config):
    return config["data"]["splits"]["train"]["expected_pairs"] * len(direction_specs(config))


def validation_layout(config):
    """Expected official validation row IDs in order: one block per direction."""
    pairs = config["data"]["splits"]["val"]["expected_pairs"]
    return [direction_row_id("val", number, position, direction)
            for position, direction in enumerate(direction_specs(config)) for number in range(1, pairs + 1)]


def balanced_indices(config, split, count):
    """First rows of each direction block, spread evenly (cache blocks are one per direction)."""
    directions = direction_specs(config)
    block = config["data"]["splits"][split]["expected_pairs"]
    per_direction = max(1, count // len(directions))
    return [position * block + index for position in range(len(directions)) for index in range(min(per_direction, block))]


def load_tokenizer(config, local_files_only=False, direction=None):
    from transformers import AutoTokenizer

    spec = config["model"]
    revision = spec["revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision) or revision != spec["tokenizer_revision"]:
        raise ValueError("Model and tokenizer require the same immutable 40-character commit revision")
    source_language = direction["source_language"] if direction else spec["source_language"]
    target_language = direction["target_language"] if direction else spec["target_language"]
    if (source_language, target_language) not in LANGUAGE_PAIRS:
        raise ValueError("This experiment requires zho_Hans -> vie_Latn or vie_Latn -> zho_Hans")
    tokenizer = AutoTokenizer.from_pretrained(
        spec["name_or_path"], revision=revision,
        src_lang=source_language, tgt_lang=target_language,
        legacy_behaviour=False,
        local_files_only=local_files_only,
    )
    for language in (source_language, target_language):
        token_id = tokenizer.convert_tokens_to_ids(language)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(f"Unknown language token: {language}")
    return tokenizer


def direction_contract(config, direction, local_files_only=False):
    """Verify one direction's language prefixes with its own tokenizer instance."""
    tokenizer = load_tokenizer(config, local_files_only, direction)
    examples = []
    pairs = (("你好，越南！", "Xin chào, Việt Nam!"), ("阮文安在河内有2026本书。", "Nguyễn Văn An có 2026 cuốn sách ở Hà Nội."))
    if direction["source_language"] == "vie_Latn":
        pairs = tuple((target, source) for source, target in pairs)
    source_id = tokenizer.convert_tokens_to_ids(direction["source_language"])
    target_id = tokenizer.convert_tokens_to_ids(direction["target_language"])
    for source, target in pairs:
        encoded = tokenizer(source, text_target=target, padding=False, truncation=False)
        if encoded["input_ids"][0] != source_id or encoded["labels"][0] != target_id:
            raise ValueError("Unexpected source/target language prefix")
        if encoded["input_ids"][-1] != tokenizer.eos_token_id or encoded["labels"][-1] != tokenizer.eos_token_id:
            raise ValueError("Expected terminal EOS")
        examples.append({"source": source, "target": target, **dict(encoded), "decoded_target": tokenizer.decode(encoded["labels"], skip_special_tokens=True)})
    return tokenizer, source_id, target_id, examples


def tokenizer_contract(config, local_files_only=False):
    from transformers import AutoConfig

    spec = config["model"]
    model_config = AutoConfig.from_pretrained(spec["name_or_path"], revision=spec["revision"], local_files_only=local_files_only)
    directions = direction_specs(config)
    contracts = []
    for direction in directions:
        tokenizer, source_id, target_id, examples = direction_contract(config, direction, local_files_only)
        contracts.append({"name": direction["name"], "source_language": direction["source_language"],
            "target_language": direction["target_language"], "source_language_id": source_id,
            "target_language_id": target_id, "forced_bos_token_id": target_id, "examples": examples})
    first = contracts[0]
    result = {
        "model": spec["name_or_path"], "revision": spec["revision"],
        "tokenizer_class": type(tokenizer).__name__, "vocabulary_size": len(tokenizer),
        "legacy_behaviour": False,
        "source_language_id": first["source_language_id"], "target_language_id": first["target_language_id"],
        "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "decoder_start_token_id": model_config.decoder_start_token_id,
        "forced_bos_token_id": first["forced_bos_token_id"],
        "label_pad_token_id": -100,
        "examples": first["examples"],
        "status": "TOKENIZER_VERIFIED_PRETRAINED_GENERATION_WAITING_FOR_COLAB",
    }
    if len(directions) > 1:
        # A shared adapter selects the target per example from the source language token.
        result["directions"] = contracts
        result["target_selection"] = "decoder prompt [decoder_start, target_language] derived from the source language token"
    return result


def cache_specification(config, inputs, source_limit, target_limit):
    if not isinstance(source_limit, int) or not isinstance(target_limit, int) or not 1 <= source_limit <= 512 or not 1 <= target_limit <= 512:
        raise ValueError("Explicit source/target token limits in [1,512] required; no truncation/filtering")
    return {
        "schema_version": 1,
        "input_manifest_sha256": inputs["manifest_sha256"],
        "splits": {name: inputs["splits"][name] for name in ("train", "val")},
        "model": config["model"], "legacy_behaviour": False,
        "versions": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "datasets", "sentencepiece")},
        "policy": {"max_source_tokens": source_limit, "max_target_tokens": target_limit, "padding": False, "truncation": False, "overflow": "fail"},
    }


def cache_key(specification):
    return sha256(json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_cache(directory, specification):
    directory = Path(directory)
    marker = directory / "cache_manifest.json"
    if not marker.is_file():
        raise ValueError("Cache incomplete: cache_manifest.json missing")
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    if manifest["cache_key"] != cache_key(specification) or manifest["specification"] != specification:
        raise ValueError("Cache fingerprint mismatch; create a separate cache")
    actual_files = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file() and p != marker}
    if actual_files != set(manifest["files"]):
        raise ValueError("Cache artifact set differs from manifest")
    for relative, entry in manifest["files"].items():
        path = contained_file(directory, relative)
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Corrupt cache artifact: {relative}")
    return manifest


def token_length_summary(lengths):
    ordered = sorted(lengths)
    if not ordered:
        raise ValueError("Empty split cannot be cached")
    return {
        "count": len(ordered),
        **{f"p{q}": ordered[math.ceil(len(ordered) * q / 100) - 1] for q in (50, 95, 99)},
        "max": ordered[-1],
        "over": {str(limit): sum(n > limit for n in ordered) for limit in (128, 256, 512)},
        "percentile_method": "nearest_rank",
    }


def prepare_cache(config, root, directory, source_limit, target_limit, local_files_only=False, batch_size=256):
    from array import array
    from datasets import Dataset, DatasetDict, load_from_disk

    if batch_size <= 0:
        raise ValueError("Tokenization batch size must be positive")
    inputs = verify_inputs(config, root)
    specification = cache_specification(config, inputs, source_limit, target_limit)
    directory = Path(directory).resolve()
    root = Path(root).resolve()
    if directory == root or root in directory.parents:
        raise ValueError("Cache must be outside frozen inputs")
    if directory.exists():
        manifest = verify_cache(directory, specification)
        restored = load_from_disk(str(directory / "dataset"))
        if any(len(restored[name]) != inputs["splits"][name]["pairs"] * len(direction_specs(config)) for name in ("train", "val")):
            raise ValueError("Restored cache count mismatch")
        return manifest
    directions = direction_specs(config)
    shared = len(directions) > 1
    tokenizers = [load_tokenizer(config, local_files_only, direction) for direction in directions]
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    lengths = {name: {"source": array("I"), "target": array("I")} for name in ("train", "val")}
    overflow_file = directory / "overflow_ids.jsonl"
    overflows = {"train": 0, "val": 0}

    def records(split):
        spec = config["data"]["splits"][split]
        # One contiguous block per direction, each with its own tokenizer (own src/tgt language prefixes).
        for position, (direction, tokenizer) in enumerate(zip(directions, tokenizers)):
            pending = []

            def encode(rows, tokenizer=tokenizer):
                tokens = tokenizer([r["source"] for r in rows], text_target=[r["target"] for r in rows], padding=False, truncation=False)
                for index, row in enumerate(rows):
                    left, right = len(tokens["input_ids"][index]), len(tokens["labels"][index])
                    lengths[split]["source"].append(left)
                    lengths[split]["target"].append(right)
                    if left > source_limit or right > target_limit:
                        overflows[split] += 1
                        overflow.write(json.dumps({"id": row["id"], "source_tokens": left, "target_tokens": right}) + "\n")
                    yield {**row, **{key: values[index] for key, values in tokens.items()}}

            names = (spec["target"], spec["source"]) if direction["swap"] else (spec["source"], spec["target"])
            for row in iter_pairs(root / names[0], root / names[1], split):
                if position:
                    row = {**row, "id": f"{row['id']}@{direction['name']}"}
                if shared:
                    row = {**row, "direction": direction["name"]}
                pending.append(row)
                if len(pending) == batch_size:
                    yield from encode(pending)
                    pending = []
            if pending:
                yield from encode(pending)

    datasets = {}
    with overflow_file.open("w", encoding="utf-8") as overflow:
        for name in ("train", "val"):
            # Disk-backed generator; no corpus-sized Python list or map copy.
            datasets[name] = Dataset.from_generator(records, gen_kwargs={"split": name},
                cache_dir=str(directory / "build"), keep_in_memory=False)
    peak_ram = None
    if platform.system() != "Windows":
        import resource
        peak_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() != "Darwin":
            peak_ram *= 1024
    audit = {"splits": {name: {side: token_length_summary(values) for side, values in sides.items()} for name, sides in lengths.items()},
        "overflows": overflows, "policy": specification["policy"], "elapsed_seconds": time.monotonic() - started,
        "peak_process_ram_bytes": peak_ram, "peak_ram_note": "ru_maxrss on Linux/macOS; unavailable on Windows",
        "references": "original full raw target text stored in Arrow", "test_tokenized": False}
    write_json(directory / "length_report.json", audit)
    if any(overflows.values()):
        raise ValueError(f"Token overflow: {overflows}; inspect length_report/overflow_ids; cache NOT complete")
    if any(len(datasets[name]) != inputs["splits"][name]["pairs"] * len(directions) for name in datasets):
        raise ValueError("Prepared cache pair count mismatch")
    DatasetDict(datasets).save_to_disk(str(directory / "dataset"))
    files = {p.relative_to(directory).as_posix(): {"bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in directory.rglob("*") if p.is_file()}
    manifest = {"cache_key": cache_key(specification), "specification": specification,
        "files": files, "disk_bytes": sum(entry["bytes"] for entry in files.values()),
        "status": "COMPLETE_CACHE", "length_report": audit}
    # Written last: interrupted/overflow cache is never treated as reusable.
    write_json(directory / "cache_manifest.json", manifest)
    verify_cache(directory, specification)
    return manifest


def contained_file(root, relative):
    root = Path(root).resolve()
    # Backslashes and drive prefixes must not become valid after crossing platforms.
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError(f"Invalid bundle relative path: {relative!r}")
    candidate = Path(relative)
    if relative.startswith("/") or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Path escapes frozen root: {relative}")
    resolved = (root / candidate).resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Path escapes frozen root: {relative}")
    if not resolved.is_file():
        raise ValueError(f"Missing input file: {relative}")
    return resolved


def iter_pairs(source, target, split):
    with Path(source).open(encoding="utf-8", newline="") as src, Path(target).open(encoding="utf-8", newline="") as tgt:
        for number, (left, right) in enumerate(zip_longest(src, tgt), 1):
            if left is None or right is None:
                raise ValueError(f"{split}:{number}: unequal source/target line counts")
            left, right = left.rstrip("\r\n"), right.rstrip("\r\n")
            if "\ufeff" in left or "\ufeff" in right:
                raise ValueError(f"{split}:{number}: BOM is not allowed")
            if not left.strip() or not right.strip():
                raise ValueError(f"{split}:{number}: empty source or target")
            yield {"id": f"{split}:{number}", "source": left, "target": right}


def verify_inputs(config, root):
    data = config["data"]
    root = Path(root).resolve()
    manifest_path = contained_file(root, data["manifest"])
    sidecar_path = contained_file(root, data["manifest_sha256_file"])
    actual = file_sha256(manifest_path)
    if actual != data["expected_manifest_sha256"] or sidecar_path.read_text(encoding="utf-8").split() != [actual, data["manifest"]]:
        raise ValueError("Frozen manifest SHA-256 does not match config pin/sidecar")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {}
    for entry in manifest["files"]:
        relative = entry["path"]
        # Validate all manifest names but require only the six selected files locally.
        if not isinstance(relative, str) or not relative or relative.startswith("/") or "\\" in relative or ":" in relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Unsafe manifest path: {relative!r}")
        normalized = Path(relative).as_posix().casefold()
        if normalized in entries:
            raise ValueError(f"Duplicate manifest path: {relative}")
        entries[normalized] = entry
    report = {"manifest_sha256": actual, "frozen_root": str(root), "splits": {}, "semantic_review": data["semantic_review"]}
    selected = set()
    for split in ("train", "val", "test"):
        spec = data["splits"][split]
        files = {}
        paths = {}
        for side in ("source", "target"):
            relative = spec[side]
            path = contained_file(root, relative)
            key = Path(relative).as_posix().casefold()
            if key in selected:
                raise ValueError(f"Input file reused across sides/splits: {relative}")
            selected.add(key)
            entry = entries.get(key)
            if entry is None:
                raise ValueError(f"Missing manifest entry: {relative}")
            digest = file_sha256(path)
            if digest != entry["sha256"] or path.stat().st_size != entry["bytes"]:
                raise ValueError(f"Input hash/size mismatch: {relative}")
            if entry["lines"] != spec["expected_pairs"]:
                raise ValueError(f"Manifest count mismatch: {relative}")
            files[side] = {"path": relative, "sha256": digest, "bytes": path.stat().st_size}
            paths[side] = path
        try:
            count = sum(1 for _ in iter_pairs(paths["source"], paths["target"], split))
        except UnicodeDecodeError as exc:
            raise ValueError(f"{split}: invalid UTF-8") from exc
        if count != spec["expected_pairs"]:
            raise ValueError(f"{split}: expected {spec['expected_pairs']} pairs, found {count}")
        report["splits"][split] = {"pairs": count, "files": files}
    report["status"] = "VERIFIED_INPUTS"
    report["usage"] = {"optimizer": "train", "selection": "val", "test": "integrity_or_final_evaluation_only"}
    return report


def copy_bundle(config, source_root, destination):
    verify_inputs(config, source_root)
    destination = Path(destination)
    source_root = Path(source_root).resolve()
    resolved = destination.resolve()
    if resolved == source_root or source_root in resolved.parents:
        raise ValueError("Bundle destination must be outside frozen source")
    # Exclusive directory creation prevents overwrite of frozen data or an existing bundle.
    destination.mkdir(parents=True, exist_ok=False)
    names = [config["data"]["manifest"], config["data"]["manifest_sha256_file"]]
    names += [spec[side] for spec in config["data"]["splits"].values() for side in ("source", "target")]
    for name in names:
        source = contained_file(source_root, name)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return verify_inputs(config, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nllb_zh_vi.json")
    parser.add_argument("--frozen-root", help="Override config frozen root, relative to current directory")
    parser.add_argument("--output", required=True, help="Verified input_manifest.json output")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify-only", action="store_true")
    action.add_argument("--bundle-output", help="Create new minimal bundle directory from frozen root")
    action.add_argument("--stage-bundle", help="Verify this bundle directory then copy to --frozen-root")
    action.add_argument("--token-contract", action="store_true", help="Inspect pinned tokenizer/config without loading model weights")
    action.add_argument("--prepare-cache", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-source-tokens", type=int)
    parser.add_argument("--max-target-tokens", type=int)
    parser.add_argument("--tokenization-batch-size", type=int, default=256)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.frozen_root or config["data"]["frozen_root"])
    output = Path(args.output).resolve()
    # Input verification must never write its report inside immutable inputs.
    protected = [root.resolve()]
    if args.bundle_output:
        protected.append(Path(args.bundle_output).resolve())
    if args.cache_dir:
        protected.append(Path(args.cache_dir).resolve())
    if args.stage_bundle:
        if args.frozen_root is None:
            parser.error("--stage-bundle requires explicit new --frozen-root destination")
        protected.append(Path(args.stage_bundle).resolve())
    if any(output == p or p in output.parents for p in protected):
        parser.error("--output must be outside source and destination input directories")
    if args.token_contract:
        report = tokenizer_contract(config, args.local_files_only)
        write_json(output, report)
        print(json.dumps({"status": report["status"], "output": str(output)}))
        return
    if args.prepare_cache:
        if not args.cache_dir:
            parser.error("--prepare-cache requires --cache-dir")
        report = prepare_cache(config, root, args.cache_dir, args.max_source_tokens, args.max_target_tokens,
                               args.local_files_only, args.tokenization_batch_size)
        write_json(output, report)
        print(json.dumps({"status": report["status"], "cache_key": report["cache_key"], "output": str(output)}))
        return
    if args.stage_bundle:
        report = copy_bundle(config, args.stage_bundle, root)
    elif args.bundle_output:
        report = copy_bundle(config, root, args.bundle_output)
    else:
        report = verify_inputs(config, root)
    write_json(output, report)
    print(json.dumps({"status": report["status"], "pairs": {k: v["pairs"] for k, v in report["splits"].items()}, "output": str(output)}))


if __name__ == "__main__":
    main()
