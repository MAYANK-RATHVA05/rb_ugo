"""Offline, non-destructive import of user-supplied research resources."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile


class ImportError(ValueError):
    """An archive is unsafe, damaged, or conflicts with existing data."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def copy_unchanged(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ImportError(f'Resource is not a file: {source}')
    if destination.exists():
        if not destination.is_file() or sha256(source) != sha256(destination):
            raise ImportError(f'Refusing to overwrite a different resource: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting an existing user file.
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    if sha256(source) != sha256(destination):
        raise ImportError(f'Copy checksum mismatch: {destination}')


def safe_extract(archive: Path, destination: Path, max_bytes: int = 512 * 1024 * 1024) -> list[Path]:
    """Check all names/CRCs/conflicts before extracting; preserve byte-identical files."""
    root = destination.resolve()
    with zipfile.ZipFile(archive) as z:
        planned = []
        keys = set()
        if sum(i.file_size for i in z.infolist()) > max_bytes:
            raise ImportError('Archive exceeds the configured uncompressed-size limit')
        for entry in z.infolist():
            name = entry.filename
            # Reject Windows paths as well as POSIX traversal; preserve original names.
            if '\\' in name or ':' in name or '\x00' in name:
                raise ImportError(f'Unsafe archive entry: {name!r}')
            rel = PurePosixPath(name)
            if rel.is_absolute() or '..' in rel.parts or not rel.parts:
                raise ImportError(f'Unsafe archive entry: {name!r}')
            mode = entry.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ImportError(f'Symlink entry is unsupported: {name!r}')
            key = str(rel).casefold()
            if key in keys:
                raise ImportError(f'Duplicate/case-colliding archive path: {name!r}')
            keys.add(key)
            target = root.joinpath(*rel.parts)
            if not target.resolve().is_relative_to(root):
                raise ImportError(f'Extraction would escape destination: {name!r}')
            if entry.is_dir():
                if target.exists() and not target.is_dir():
                    raise ImportError(f'Directory conflicts with existing file: {target}')
                continue
            # An existing parent file would make extraction incomplete.
            if any(p.exists() and not p.is_dir() for p in target.parents if p.is_relative_to(root)):
                raise ImportError(f'Parent path conflicts with existing file: {target}')
            data = z.read(entry)  # zipfile validates each CRC during read.
            if target.exists() and (not target.is_file() or target.read_bytes() != data):
                raise ImportError(f'Refusing to overwrite extracted user data: {target}')
            planned.append((target, data))
        file_targets = {target for target, _ in planned}
        if any(parent in file_targets for target in file_targets for parent in target.parents):
            raise ImportError('Archive file/directory path conflict')
        for target, data in planned:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with target.open('xb') as stream:
                    stream.write(data)
            if target.read_bytes() != data:
                raise ImportError(f'Extraction verification failed: {target}')
    return [p for p, _ in planned]


def import_dataset(archive: Path, name: str, root: Path) -> list[Path]:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ImportError('Dataset name must be a simple directory name')
    paths = safe_extract(archive, root / 'data/raw/keel' / name)
    copy_unchanged(archive, root / 'data/incoming' / archive.name)
    return paths


def assignment(value: str) -> tuple[str, Path]:
    if '=' not in value:
        raise argparse.ArgumentTypeError('Expected NAME=PATH')
    name, path = value.split('=', 1)
    return name, Path(path).expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--dataset-zip', type=assignment, action='append', default=[], metavar='NAME=PATH')
    parser.add_argument('--paper-file', type=assignment, action='append', default=[], metavar='CATEGORY=PATH')
    parser.add_argument('--paper-zip', type=Path, action='append', default=[], help='Extract a supplied related-paper collection')
    args = parser.parse_args()
    try:
        for name, archive in args.dataset_zip:
            files = import_dataset(archive, name, args.root)
            print(f'{name}: verified {len(files)} extracted files; ZIP unchanged')
        for category, source in args.paper_file:
            if category not in {'base', 'related', 'additional'}:
                raise ImportError('Paper category must be base, related or additional')
            copy_unchanged(source, args.root / 'papers' / category / source.name)
            print(f'Paper copied unchanged: {source.name}')
        for archive in args.paper_zip:
            files = safe_extract(archive, args.root / 'papers/related' / archive.stem)
            copy_unchanged(archive, args.root / 'data/incoming' / archive.name)
            print(f'Paper collection: {len(files)} verified files')
    except (ImportError, OSError, zipfile.BadZipFile, RuntimeError) as exc:
        parser.exit(2, f'Import failed: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
