"""Isolated removal of one unnecessary TRL token-accuracy logits copy.

The Python method comes from TRL 0.29.1 (Apache-2.0):
https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py
The sole AST change is the ordinary causal accuracy assignment:
    outputs.logits[..., :-1, :].contiguous() -> outputs.logits[..., :-1, :]
PyTorch argmax accepts that strided view. All branches, masks, gather calls,
outputs, and losses remain the original method's operations. The clone runs
only inside private_logits' guarded training scope; all other calls delegate.
"""

import ast
import copy
import functools
import hashlib
import importlib.metadata
import inspect
import textwrap
from types import FunctionType


TRL_VERSION = "0.29.1"
SOURCE_SHA256 = "2b060ef7c0acc470d38c72a7a2e468a0d66e7f33a9008c947522245943723562"
_LAST_GUARD_REASON = "not installed"


def _expression(source):
    return ast.parse(source, mode="eval").body


def _same(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def _clone_pinned_method(original, owner):
    """Clone one AST edit while retaining zero-argument super's original cell."""
    source = textwrap.dedent(inspect.getsource(original)).replace("\r\n", "\n")
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != SOURCE_SHA256:
        raise ValueError("TRL compute_loss source fingerprint differs")
    freevars = original.__code__.co_freevars
    if (freevars != ("__class__",) or not original.__closure__
            or original.__closure__[0].cell_contents is not owner):
        raise ValueError("unexpected compute_loss class closure")
    tree = ast.parse(source)
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef) or function.decorator_list:
        raise ValueError("unexpected compute_loss definition")

    # The only loads of shift_logits are its optional prompt-token re-slice
    # and argmax. Neither needs contiguous storage. No .view, reshape, external
    # call, in-place write, or public return consumes this temporary variable.
    parents = {child: parent for parent in ast.walk(function)
               for child in ast.iter_child_nodes(parent)}
    expected_slice = _expression("shift_logits[:, self.num_virtual_tokens :, :]")
    loads = [node for node in ast.walk(function)
             if isinstance(node, ast.Name) and node.id == "shift_logits"
             and isinstance(node.ctx, ast.Load)]
    if len(loads) != 2:
        raise ValueError("unexpected shift_logits consumers")
    for node in loads:
        parent = parents[node]
        is_argmax = (isinstance(parent, ast.Attribute) and parent.value is node
                     and parent.attr == "argmax"
                     and isinstance(parents.get(parent), ast.Call))
        is_prompt_slice = isinstance(parent, ast.Subscript) and _same(parent, expected_slice)
        if not (is_argmax or is_prompt_slice):
            raise ValueError("shift_logits gained a consumer requiring review")

    modified = copy.deepcopy(function)
    expected_copy = _expression("outputs.logits[..., :-1, :].contiguous()")
    edits = 0
    for node in ast.walk(modified):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "shift_logits"
                and _same(node.value, expected_copy)):
            node.value = ast.copy_location(node.value.func.value, node.value)
            edits += 1
    if edits != 1:
        raise ValueError("expected exactly one ordinary accuracy copy")

    # Compiling inside a temporary class preserves the __class__ freevar that
    # Python gives the original zero-argument super(). Rebind that code to the
    # ORIGINAL closure and module globals, not the temporary class or a copied
    # globals snapshot. Future entropy alias changes are therefore still seen.
    container = ast.ClassDef(name="_StridedAccuracyClone", bases=[], keywords=[],
                             body=[modified], decorator_list=[], type_params=[])
    module = ast.fix_missing_locations(ast.Module(body=[container], type_ignores=[]))
    temporary = dict(original.__globals__)
    exec(compile(module, "<soup-strided-accuracy-clone>", "exec"), temporary)
    code = temporary[container.name].__dict__[original.__name__].__code__
    if code.co_freevars != freevars:
        raise ValueError("compiled class closure changed")
    result = FunctionType(code, original.__globals__, original.__name__,
                          original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    functools.update_wrapper(result, original)
    result._soup_removed_accuracy_copy = True
    return result


def guard_reason():
    return _LAST_GUARD_REASON


def install():
    """Fail closed on another TRL version/source; install no global Tensor hook."""
    global _LAST_GUARD_REASON
    from trl import SFTTrainer
    from .private_logits import private_logits_active

    original = SFTTrainer.compute_loss
    if getattr(original, "_soup_strided_accuracy_patch", False):
        _LAST_GUARD_REASON = "already installed"
        return False
    if importlib.metadata.version("trl") != TRL_VERSION:
        _LAST_GUARD_REASON = "TRL version differs"
        return False
    try:
        cloned = _clone_pinned_method(original, SFTTrainer)
    except (OSError, TypeError, ValueError) as error:
        _LAST_GUARD_REASON = str(error)
        return False

    @functools.wraps(original)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        function = cloned if private_logits_active() else original
        return function(self, model, inputs, return_outputs=return_outputs,
                        num_items_in_batch=num_items_in_batch)

    compute_loss._soup_strided_accuracy_patch = True
    SFTTrainer.compute_loss = compute_loss
    _LAST_GUARD_REASON = "installed; one accuracy copy removed only in private scope"
    return True
