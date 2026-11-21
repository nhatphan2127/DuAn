"""Build the Colab source ZIP (the files in CODE_FILES plus SOURCE_MANIFEST.json) reproducibly."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from nllb_lora.train.colab_workflow import CODE_FILES


def build_bundle(root, output):
    root, output = Path(root), Path(output)
    missing = [name for name in CODE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing code files in " + str(root) + ":\n" + "\n".join(missing))
    payloads = {name: (root / name).read_bytes() for name in CODE_FILES}
    manifest = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in payloads.items()}
    payloads["SOURCE_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stored (not deflated) with fixed timestamps so the ZIP hash does not depend on the zlib version.
    with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED) as bundle:
        for name, data in payloads.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            info.create_system = 3  # zipfile defaults to the host OS (0 on Windows); fix it so the hash is OS-independent
            bundle.writestr(info, data)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + ".sha256").write_text(digest + "  " + output.name + "\n", encoding="utf-8")
    return digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="dist/nllb_lora_colab_source.zip")
    args = parser.parse_args()
    print(build_bundle(args.root, args.output), args.output)


if __name__ == "__main__":
    main()
