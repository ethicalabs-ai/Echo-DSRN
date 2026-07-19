#!/usr/bin/env python3
"""
scripts/sync_hf_modules.py
────────────────────────────────────────────────────────────────────────────
Keep custom Python modules on Hugging Face repos in sync with local source.

Reads model_registry.yaml, downloads each checkpoint, copies current .py
modules from the local source tree, patches imports for HF compatibility,
tests model loading, and uploads back to the Hub.

Usage
─────
    uv run --extra rocm python scripts/sync_hf_modules.py              # sync all
    uv run --extra rocm python scripts/sync_hf_modules.py --dry-run    # preview only
    uv run --extra rocm python scripts/sync_hf_modules.py --model ethicalabs/Echo-DSRN-114M-v0.1.2
    uv run --extra rocm python scripts/sync_hf_modules.py --skip-upload
    uv run --extra rocm python scripts/sync_hf_modules.py --skip-test
"""

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import click
import yaml

# ---------------------------------------------------------------------------
# Project root (where echo_dsrn/ and echo_hybrid/ live)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "model_registry.yaml"


# ---------------------------------------------------------------------------
# TYPE_CHECKING patches — force HF dynamic module loader to bundle
# transitive .py dependencies (triton_scan.py, utils.py).
#
# Without these, get_relative_imports fails with FileNotFoundError.
# ---------------------------------------------------------------------------
TYPE_CHECKING_PATCHES = {
    "modeling_echo.py": (
        "if TYPE_CHECKING:\n    # Force HF trust_remote_code AST parser to bundle triton_scan.py\n    pass",
        "if TYPE_CHECKING:\n    # Force HF trust_remote_code AST parser to bundle triton_scan.py and utils.py\n    from .triton_scan import triton_dsrn_parallel_scan\n    from .utils import rms_norm_fn",
    ),
    "modeling_generative_clf.py": (
        "if typing.TYPE_CHECKING:\n    # Force HF trust_remote_code to bundle nested dependencies\n    pass",
        "if typing.TYPE_CHECKING:\n    # Force HF trust_remote_code to bundle nested dependencies\n    from .triton_scan import triton_dsrn_parallel_scan\n    from .utils import rms_norm_fn",
    ),
}

# Import prefix replacements (absolute → relative, one level deep)
IMPORT_REPLACEMENTS = [
    ("from echo_dsrn.", "from ."),
    ("from echo_hybrid.", "from ."),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_file(path: Path) -> str:
    """SHA-256 of file contents, for change detection."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _patch_py_file(path: Path) -> int:
    """Apply import replacements and TYPE_CHECKING patches to a .py file.

    Returns number of patches applied.
    """
    content = path.read_text()
    original = content

    # Absolute → relative imports
    for old, new in IMPORT_REPLACEMENTS:
        content = content.replace(old, new)

    # TYPE_CHECKING blocks
    fname = path.name
    if fname in TYPE_CHECKING_PATCHES:
        old_block, new_block = TYPE_CHECKING_PATCHES[fname]
        if old_block in content:
            content = content.replace(old_block, new_block)

    # __init__.py: remove generative-CLF import to avoid circular imports.
    # In the HF checkpoint context, __init__.py is loaded as the package init
    # when any module does a relative import.  The generative-CLF import
    # triggers modeling_generative_clf → modeling_echo → __init__ (cycle).
    if fname == "__init__.py":
        content = _strip_init_generative_clf(content)

    if content == original:
        return 0

    path.write_text(content)
    return 1


def _strip_init_generative_clf(content: str) -> str:
    """Remove EchoForGenerativeClassification from __init__.py.

    This import causes a circular import when HF's dynamic module loader
    processes the flat checkpoint directory as a package.
    """
    lines = content.split("\n")
    out = []
    for line in lines:
        stripped = line.strip()
        # Remove the import line
        if "from .modeling_generative_clf import" in stripped:
            continue
        # Remove it from __all__
        if stripped == '"EchoForGenerativeClassification",':
            continue
        out.append(line)
    return "\n".join(out)


def _copy_modules(sources: list[dict], dst_dir: Path, dry_run: bool) -> list[str]:
    """Copy .py files from source directories to dst_dir. Returns list of copied names."""
    copied = []
    for src_entry in sources:
        src_dir = PROJECT_ROOT / src_entry["dir"]
        for fname in src_entry["files"]:
            src = src_dir / fname
            if not src.exists():
                raise FileNotFoundError(f"Source file missing: {src}")
            dst = dst_dir / fname
            if not dry_run:
                shutil.copy2(src, dst)
            copied.append(fname)
    return copied


def _patch_all(dst_dir: Path, dry_run: bool) -> int:
    """Patch all .py files in dst_dir. Returns count of patched files."""
    count = 0
    for fname in sorted(os.listdir(dst_dir)):
        if not fname.endswith(".py"):
            continue
        if not dry_run:
            count += _patch_py_file(dst_dir / fname)
    return count


def _needs_sync(repo_id: str, work_dir: Path, sources: list[dict]) -> bool:
    """Check whether any source file differs from the HF copy."""
    for src_entry in sources:
        src_dir = PROJECT_ROOT / src_entry["dir"]
        for fname in src_entry["files"]:
            local_hash = _hash_file(src_dir / fname)
            remote = work_dir / fname
            if not remote.exists():
                return True
            remote_hash = _hash_file(remote)
            if local_hash != remote_hash:
                return True
    return False


def _test_model(work_dir: Path, auto_class: str) -> Optional[str]:
    """Try loading the model from work_dir. Returns None on success, error str on failure."""

    code = f"""
import sys
from transformers import {auto_class}

repo = {str(work_dir)!r}

# Pre-cache all .py files to work around HF dynamic-module-loader bugs
# with transitive relative imports (hybrid models especially).
try:
    from transformers.dynamic_module_utils import get_cached_module_file
    import os
    for f in os.listdir(repo):
        if f.endswith('.py'):
            try:
                get_cached_module_file(repo, f)
            except Exception:
                pass
except Exception:
    pass

m = {auto_class}.from_pretrained(repo, trust_remote_code=True, dtype='bfloat16')
name = type(m).__name__
params = sum(p.numel() for p in m.parameters())
print(f"OK  {{name}}  {{params/1e6:.1f}}M params")
sys.exit(0)
"""
    result = subprocess.run(
        ["uv", "run", "--extra", "rocm", "python3", "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    if result.returncode == 0:
        return None
    # Extract last meaningful line from stderr
    return result.stderr.strip().split("\n")[-1] if result.stderr else result.stdout.strip()


def _upload(repo_id: str, work_dir: Path) -> subprocess.CompletedProcess:
    """Upload work_dir to HF repo via `hf upload`."""
    return subprocess.run(
        ["uv", "run", "--extra", "rocm", "hf", "upload", repo_id, str(work_dir)],
        capture_output=True,
        text=True,
        timeout=300,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--model",
    "-m",
    multiple=True,
    help="Sync only specific model(s). Repeatable. Omit to sync all.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without making changes.",
)
@click.option(
    "--skip-test",
    is_flag=True,
    help="Skip the model-loading verification step.",
)
@click.option(
    "--skip-upload",
    is_flag=True,
    help="Prepare checkpoints but do not push to HF.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Sync even if no diff is detected.",
)
def main(model, dry_run, skip_test, skip_upload, force):
    """Sync custom Python modules from local source to HF repos."""
    if not REGISTRY_PATH.exists():
        click.secho(f"ERROR: Registry not found: {REGISTRY_PATH}", fg="red")
        raise SystemExit(1)

    with open(REGISTRY_PATH) as f:
        registry = yaml.safe_load(f)

    models = registry.get("models", {})
    if not models:
        click.secho("No models in registry.", fg="yellow")
        return

    # Filter by --model
    if model:
        selected = {m: models[m] for m in model if m in models}
        missing = set(model) - set(models.keys())
        if missing:
            click.secho(f"WARNING: Models not in registry: {', '.join(missing)}", fg="yellow")
        models = selected

    if not models:
        click.secho("No models to sync.", fg="yellow")
        return

    action = "DRY-RUN" if dry_run else "SYNC"
    click.secho(f"{action}: {len(models)} model(s)\n", fg="cyan", bold=True)

    results = {}
    for repo_id, cfg in models.items():
        click.secho(f"── {repo_id} ──", fg="blue", bold=True)
        auto_class = cfg["auto_class"]
        sources = cfg["sources"]

        # Use a repo-specific directory name to avoid HF dynamic-module-
        # loader cache collisions (HF caches by directory basename).
        safe_name = repo_id.replace("/", "_").replace("-", "_").replace(".", "_")
        with tempfile.TemporaryDirectory(prefix="hf-sync-") as tmp:
            work_dir = Path(tmp) / safe_name
            work_dir.mkdir()

            # 1. Download
            click.echo("  Downloading …", nl=False)
            if not dry_run:
                subprocess.run(
                    [
                        "uv",
                        "run",
                        "--extra",
                        "rocm",
                        "hf",
                        "download",
                        repo_id,
                        "--local-dir",
                        str(work_dir),
                    ],
                    capture_output=True,
                    check=True,
                )
            click.secho(" ✓", fg="green")

            # 2. Check if sync needed
            if not force and not _needs_sync(repo_id, work_dir, sources):
                click.secho("  Skipped — modules already in sync.", fg="yellow")
                results[repo_id] = "skip"
                continue

            # 3. Copy modules
            copied = _copy_modules(sources, work_dir, dry_run)
            click.echo(f"  Copied {len(copied)} module(s): {', '.join(copied)}")

            # 4. Patch
            patched = _patch_all(work_dir, dry_run)
            if patched:
                click.echo(f"  Patched {patched} file(s)")

            # 5. Test
            if not skip_test:
                click.echo(f"  Testing load as {auto_class} …", nl=False)
                if dry_run:
                    click.secho(" (skipped in dry-run)", fg="yellow")
                else:
                    error = _test_model(work_dir, auto_class)
                    if error is None:
                        click.secho(" ✓", fg="green")
                    else:
                        click.secho(f" ✗ {error}", fg="red")
                        results[repo_id] = "test-failed"
                        continue

            # 6. Upload
            if not skip_upload:
                click.echo("  Uploading …", nl=False)
                if dry_run:
                    click.secho(" (skipped in dry-run)", fg="yellow")
                else:
                    result = _upload(repo_id, work_dir)
                    if result.returncode == 0:
                        click.secho(" ✓", fg="green")
                        results[repo_id] = "uploaded"
                    else:
                        click.secho(f" ✗ {result.stderr.strip()[-120:]}", fg="red")
                        results[repo_id] = "upload-failed"
            else:
                click.secho(f"  Ready at {work_dir} (upload skipped)", fg="yellow")
                results[repo_id] = "ready"

    # Summary
    click.echo()
    for repo_id, status in results.items():
        color = {
            "uploaded": "green",
            "ready": "yellow",
            "skip": "yellow",
            "test-failed": "red",
            "upload-failed": "red",
        }.get(status, "white")
        click.secho(f"  {status:15s} {repo_id}", fg=color)


if __name__ == "__main__":
    main()
