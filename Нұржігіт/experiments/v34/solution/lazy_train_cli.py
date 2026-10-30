"""Load only the requested train command in the pinned Soup CLI.

Keep the original train function, root callback, error/audit/telemetry helpers
and run() unchanged. Unrelated command modules and registrations are deferred.
Other commands, an already imported CLI, or another source revision use the
ordinary complete CLI. No training work or cache is moved outside the process.
"""
import ast
import hashlib
import importlib.abc
import importlib.machinery
import sys

# SourceLoader.get_source normalizes newlines. Exact soup-cli 0.77.0 source.
CLI_SHA256 = '90f9dcdcea2aebf3159e5fc4b7aa7e9343b696b14563b4d7fd2a0ef590721479'


def train_only_source(source, filename):
    """Return compiled, source-guarded CLI and diagnostics, or decline to alter it."""
    if hashlib.sha256(source.encode('utf-8')).hexdigest() != CLI_SHA256:
        return None
    tree = ast.parse(source, filename=filename)
    definitions = {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    blocked = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('soup_cli.commands'):
            blocked.update(a.asname or a.name for a in node.names
                           if not (node.module == 'soup_cli.commands' and a.name == 'train'))
    # No retained function may depend on a module whose eager import is removed.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if any(isinstance(n, ast.Name) and n.id in blocked for n in ast.walk(node)):
                return None
    kept, train_registration = [], 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('soup_cli.commands'):
            names = [a for a in node.names if node.module == 'soup_cli.commands' and a.name == 'train']
            if names:
                node.names = names
                kept.append(node)
            continue
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if any(isinstance(n, ast.Name) and n.id in blocked for n in ast.walk(node)):
                continue
            # Inline empty command groups do not need registration either.
            registration = (isinstance(node, ast.Expr) and any(
                isinstance(n, ast.Attribute) and n.attr in ('command', 'add_typer')
                for n in ast.walk(node)))
            if registration:
                is_train = any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                               and n.value.id == 'train' and n.attr == 'train' for n in ast.walk(node))
                if not is_train:
                    continue
                train_registration += 1
        kept.append(node)
    tree.body = kept
    after = {n.name: ast.dump(n, include_attributes=False) for n in kept
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    if definitions != after or not {'main', 'run'} <= definitions.keys() or train_registration != 1:
        return None
    return compile(ast.fix_missing_locations(tree), filename, 'exec', dont_inherit=True), {
        'original_source_sha256': CLI_SHA256, 'deferred_command_bindings': len(blocked),
        'original_function_definitions_preserved': len(definitions), 'train_registrations': 1}


class _TrainLoader(importlib.abc.Loader):
    def __init__(self, delegate, code, info):
        self.delegate, self.code, self.info = delegate, code, info

    def create_module(self, spec):
        return self.delegate.create_module(spec)

    def exec_module(self, module):
        exec(self.code, module.__dict__)
        module.__solution_lazy_cli_info__ = self.info
        print('solution: train CLI deferred '
              f"{self.info['deferred_command_bindings']} unrelated command bindings", flush=True)

    def get_source(self, fullname):
        return self.delegate.get_source(fullname)

    def get_filename(self, fullname):
        return self.delegate.get_filename(fullname)


class _TrainFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'soup_cli.cli':
            return None
        # This single-use hook does not affect other module imports.
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None or not hasattr(spec.loader, 'get_source'):
            return None
        source = spec.loader.get_source(fullname)
        prepared = train_only_source(source, spec.origin) if source is not None else None
        if prepared is None:
            return None
        code, info = prepared
        spec.loader = _TrainLoader(spec.loader, code, info)
        return spec


def patch_lazy_train_cli():
    if (len(sys.argv) < 2 or sys.argv[1] != 'train' or 'soup_cli.cli' in sys.modules or
            any(isinstance(finder, _TrainFinder) for finder in sys.meta_path)):
        return False
    sys.meta_path.insert(0, _TrainFinder())
    return True
