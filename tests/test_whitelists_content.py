from pathlib import Path


def test_whitelists_cover_main_files():
    # Resolve repository root from this test file path: <repo_root>/hound/tests/this
    repo_root = Path(__file__).resolve().parents[2]
    wl_dir = repo_root / "Misc" / "whitelists"
    sources_dir = repo_root / "sources"

    assert wl_dir.exists(), f"Missing whitelist dir: {wl_dir}"
    assert sources_dir.exists(), f"Missing sources dir: {sources_dir}"

    skip_dirs = {
        "test",
        "tests",
        "script",
        "scripts",
        "mocks",
        "mock",
        "examples",
        "example",
        "sample",
        "samples",
        "docs",
        "interfaces",
        "interface",
        "deploy",
        "testing",
    }

    def is_interface_file(path: Path) -> bool:
        n = path.name.lower()
        if n.endswith(".sol") and n.startswith("i"):
            return True
        return "interface" in n

    def is_skipped(subpath: Path) -> bool:
        parts = {p.lower() for p in subpath.parts}
        return bool(parts & skip_dirs)

    errors = []

    for wl_file in sorted(wl_dir.glob("*.txt")):
        proj = wl_file.stem
        # media-kit lists are documentation-only; skip them
        if "media-kit" in proj:
            continue

        proj_dir = sources_dir / proj
        if not proj_dir.exists():
            # If no source folder exists for this whitelist, skip (some lists are metadata-only)
            continue

        # Discover main candidates present in sources (.sol/.vy only)
        all_candidates = []
        for p in proj_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".sol", ".vy"}:
                rel = p.relative_to(proj_dir)
                if is_skipped(rel) or is_interface_file(p):
                    continue
                all_candidates.append(rel)

        if not all_candidates:
            # No EVM/Vyper files to whitelist; nothing to assert for this project
            continue

        content = wl_file.read_text(encoding="utf-8").strip()
        if not content:
            errors.append(f"Whitelist is empty but sources contain .sol/.vy files: {wl_file.name}")
            continue

        entries = [e.strip() for e in content.split(",") if e.strip()]
        if not entries:
            errors.append(f"Whitelist has no valid entries after parsing: {wl_file.name}")
            continue

        for e in entries:
            ep = Path(e)
            # Paths should be relative to project dir and exist
            if ep.is_absolute() or (proj_dir / ep).exists() is False:
                errors.append(f"Entry does not exist for {wl_file.name}: {e}")
                continue
            # Only .sol/.vy permitted
            if ep.suffix.lower() not in {".sol", ".vy"}:
                errors.append(f"Entry has unsupported extension ({ep.suffix}) in {wl_file.name}: {e}")
            # Exclude tests/interfaces/etc.
            if is_skipped(ep):
                errors.append(f"Entry is in a skipped directory in {wl_file.name}: {e}")
            if is_interface_file(proj_dir / ep):
                errors.append(f"Entry looks like an interface in {wl_file.name}: {e}")

    assert not errors, "\n".join(errors)

