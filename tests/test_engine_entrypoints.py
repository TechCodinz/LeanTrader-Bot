"""Every engine method the orchestrator schedules must exist on that engine.

Eight scheduled loops used to await a method their engine did not define --
execute_scalp_trades, scan_arbitrage, hunt_micro_moons, trade_forex_pairs,
execute_continuous_trading, train_models, scan_all_platforms,
run_continuous_trading. Each raised AttributeError on every pass into a
``logger.debug`` handler and slept, so the failure never appeared at INFO,
while startup had already logged the engine as ACTIVE.

This is a static check, so it runs without constructing engines or touching
the network, and it fails on a renamed method the moment it is introduced.
"""

import ast
import collections
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
ORCHESTRATOR = ROOT / "COMPLETE_ULTIMATE_ORCHESTRATOR.py"

# Orchestrator files that assign engines onto self.
ASSIGNMENT_SOURCES = (
    "COMPLETE_ULTIMATE_ORCHESTRATOR.py",
    "ULTIMATE_ORCHESTRATOR.py",
    "COMPLETE_UNIFIED_ORCHESTRATOR.py",
)


def _class_name_of(node):
    """The class a call expression constructs, if it is a plain constructor."""
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
    if isinstance(node, ast.BoolOp):
        # self.x = registry.get("x") or SomeEngine(...)
        for value in node.values:
            name = _class_name_of(value)
            if name:
                return name
    return None


@pytest.fixture(scope="module")
def attribute_classes():
    """self.<attr> -> the class it is constructed from."""
    mapping = {}
    for filename in ASSIGNMENT_SOURCES:
        path = ROOT / filename
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(errors="ignore"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            class_name = _class_name_of(node.value)
            if not class_name:
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    mapping.setdefault(target.attr, class_name)
    return mapping


@pytest.fixture(scope="module")
def class_index():
    """class -> (its own methods, its base class names), repo wide."""
    methods = collections.defaultdict(set)
    bases = {}
    for path in ROOT.rglob("*.py"):
        if any(part in {".git", "__pycache__", "node_modules", "build"} for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods[node.name].add(item.name)
            bases.setdefault(
                node.name, [b.id for b in node.bases if isinstance(b, ast.Name)]
            )
    return methods, bases


def _resolve_methods(class_name, index, seen=None):
    methods, bases = index
    seen = seen or set()
    if class_name in seen:
        return set()
    seen.add(class_name)
    found = set(methods.get(class_name, ()))
    for base in bases.get(class_name, []):
        found |= _resolve_methods(base, index, seen)
    return found


def _engine_calls():
    """Every self.<attr>.<method>(...) call site in the orchestrator."""
    tree = ast.parse(ORCHESTRATOR.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "self"
        ):
            continue
        yield node.lineno, func.value.attr, func.attr


def test_the_orchestrator_module_parses():
    assert ORCHESTRATOR.exists()
    ast.parse(ORCHESTRATOR.read_text())


def test_no_scheduled_call_targets_a_method_its_engine_does_not_define(
    attribute_classes, class_index
):
    missing = []
    for lineno, attr, method in _engine_calls():
        class_name = attribute_classes.get(attr)
        if not class_name:
            continue
        known = _resolve_methods(class_name, class_index)
        if not known:
            # Class not found in this tree (third-party or dynamic); nothing
            # to check it against.
            continue
        if method not in known:
            missing.append(
                f"{ORCHESTRATOR.name}:{lineno} self.{attr} is a {class_name}, "
                f"which defines no {method}()"
            )

    assert not missing, "orchestrator calls methods that do not exist:\n  " + "\n  ".join(
        sorted(set(missing))
    )


@pytest.mark.parametrize(
    "class_name,method",
    [
        ("UltraScalpingEngine", "start_scalping"),
        ("UltraArbitrageEngine", "start_arbitrage_scanning"),
        ("UltraContinuousTradingOrchestrator", "start_continuous_trading"),
        ("UltraMultiPlatformScanner", "start_multi_platform_scanning"),
        ("UltraMoonSystem", "run_forever"),
        ("UltraMLPipeline", "run_forever"),
        ("ContinuousUltraTradingSystem", "start"),
    ],
)
def test_the_entrypoints_the_orchestrator_now_uses_are_real(
    class_index, class_name, method
):
    assert method in _resolve_methods(class_name, class_index)


def test_the_continuous_orchestrator_can_adopt_canonical_engines(class_index):
    """It starts scalping and arbitrage, so it must not own private copies.

    Its constructor built its own UltraScalpingEngine and UltraArbitrageEngine
    while the top-level orchestrator supervised different instances of the
    same classes. The start-once guards are per-instance, so that meant two of
    each trading the same account.
    """
    assert "adopt_engines" in _resolve_methods(
        "UltraContinuousTradingOrchestrator", class_index
    )

    source = (ROOT / "COMPLETE_ULTIMATE_ORCHESTRATOR.py").read_text()
    assert "adopt_engines(" in source, (
        "the orchestrator must hand its canonical engines to "
        "UltraContinuousTradingOrchestrator before it starts them"
    )


def test_the_dex_sniper_does_not_fabricate_a_transaction_hash():
    """It returned a sha256 of the unsigned tx dict as a tx_hash."""
    source = (ROOT / "ultra_moon_spotter.py").read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_execute_snipe":
            body = ast.get_source_segment(source, node) or ""
            assert "sha256" not in body, "a hashed tx dict is not a receipt"
            assert "'success': True" not in body
            assert "dex_snipe_signing_not_configured" in body
            break
    else:
        pytest.fail("_execute_snipe not found")
