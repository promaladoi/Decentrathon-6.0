"""Infer installed package names without stat'ing every installed package file.

CPython 3.13's metadata scan reads every RECORD, constructs every PackagePath,
and checks every file exists before deduplicating its top-level package names.
We keep the same RECORD and existence semantics but stop checking a package once
one of its recorded files exists. No availability result or cross-run cache is
invented. Non-filesystem/custom distributions keep the ordinary implementation.
"""
import csv
import importlib.metadata as metadata
from pathlib import Path
import sys


def record_names(dist, lines, get_name):
    found = set()
    for row in csv.reader(lines):
        # Preserve the standard reader's validation, including files whose name
        # cannot become a package. Let the original implementation handle errors.
        if not 1 <= len(row) <= 3:
            raise ValueError('unsupported RECORD row')
        if len(row) == 3 and row[2]:
            int(row[2])
        name = row[0]
        prefix, sep, _rest = name.partition('/')
        if sep and prefix not in ('', '.', '..'):
            top = prefix
        else:
            top = get_name(metadata.PackagePath(name))
        if top is None or '.' in top or top in found:
            continue
        if dist.locate_file(name).exists():
            found.add(top)
    return found


def make_inferred(original, get_name):
    def inferred(dist):
        if (type(dist) is not metadata.PathDistribution or
                not isinstance(getattr(dist, '_path', None), Path) or
                metadata.PathDistribution.files is not metadata.Distribution.files):
            return original(dist)
        lines = dist._read_files_distinfo()
        if not lines:
            return original(dist)
        try:
            return record_names(dist, lines, get_name)
        except (ValueError, TypeError, IndexError, OSError, csv.Error):
            return original(dist)
    inferred._solution_record_scan = True
    return inferred


def patch_metadata_scan():
    if (sys.version_info[:2] != (3, 13) or
            not hasattr(metadata, '_get_toplevel_name')):
        return False
    original = metadata._top_level_inferred
    if getattr(original, '_solution_record_scan', False):
        return True
    # Only the inspected CPython implementation, not an unrelated installed hook.
    if (original.__module__ != 'importlib.metadata' or
            set(original.__code__.co_names) !=
            {'set', 'map', '_get_toplevel_name', 'always_iterable', 'files', 'filter'}):
        return False
    metadata._top_level_inferred = make_inferred(original, metadata._get_toplevel_name)
    return True
