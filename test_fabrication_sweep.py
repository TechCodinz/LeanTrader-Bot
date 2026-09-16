"""No module the bot can reach may invent market data.

This is the invariant the whole sweep exists to protect. It is written
against the import closure rather than a file list, so a future import that
drags fabricated data into the trading path fails here rather than in
production.

Randomness that is correct is deliberately not flagged: neural-net weight
initialisation, particle-swarm starting positions, epsilon-greedy bandit
exploration, and the paper broker and emulator whose job is to simulate.
"""

import ast
import subprocess
import sys

import pytest


def sites(path):
    tree = ast.parse(open(path, errors="replace").read())
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"uniform", "choice", "random", "randint", "randn", "normal", "gauss"}
        and _base_of(n.func.value) in {"np", "random"}
    ]


def _base_of(node):
    """The root name of a call target.

    np.random.randn is an Attribute whose value is another Attribute, so a
    plain isinstance(..., ast.Name) check silently misses every numpy draw --
    which is most of them.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def test_the_audit_gate_passes_for_the_trading_path():
    result = subprocess.run(
        [sys.executable, "tools/fabrication_audit.py"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout
    assert "PASS" in result.stdout


@pytest.mark.parametrize(
    "path",
    [
        "ultra_scout.py",
        "ultra_swarm_consciousness.py",
        "ultra_fluid_mechanics.py",
        "ultra_multi_platform_scanner.py",
        "ml_strategy_engine.py",
    ],
)
def test_the_repaired_engines_no_longer_fabricate(path):
    assert sites(path) == [], f"{path} still invents data"


def test_monte_carlo_var_keeps_its_draw():
    """Drawing samples IS Monte Carlo. Only its inputs must be real.

    nobel_risk_management keeps exactly two draws: the VaR simulation, whose
    mean and standard deviation come from real returns, and a __main__ demo
    fixture. Everything that fed fabricated data into a risk figure is gone.
    """
    src = open("nobel_risk_management.py").read()
    assert "simulations = np.random.normal(mean, std, n_simulations)" in src
    # The two that did fabricate are gone.
    assert "For now, return random returns" not in src
    assert "return 0.05  # 5% daily risk" not in src

    # Only the simulation and the demo remain.
    assert len(sites("nobel_risk_management.py")) == 2


def test_the_swarm_reads_real_candles_or_none():
    src = open("ultra_swarm_consciousness.py").read()
    assert "exchange.fetch_ohlcv(symbol" in src
    assert "Simulate market data fetching" not in src


def test_fluid_mechanics_is_no_longer_hardcoded_to_fifty_thousand():
    src = open("ultra_fluid_mechanics.py").read()
    assert "base_price = 50000" not in src
    assert "fetch_ohlcv" in src


def test_arbitrage_uses_executable_prices_from_two_real_venues():
    """It drew a random price per platform and called the spread arbitrage."""
    src = open("ultra_multi_platform_scanner.py").read()
    # Parsed, not grepped: the repair's own comment names the call it removed.
    assert sites("ultra_multi_platform_scanner.py") == []
    assert "if len(platform_prices) < 2:" in src
    # Buy at the ask, sell at the bid -- the prices actually executable.
    assert 'kv[1]["ask"]' in src and 'kv[1]["bid"]' in src


def test_unconfigured_scanners_return_nothing():
    """No adapter means no opportunities, not invented ones."""
    src = open("ultra_multi_platform_scanner.py").read()
    for adapter in ("defi_adapter", "yield_adapter", "platform_adapter", "dex_adapter"):
        assert adapter in src


def test_legitimate_randomness_is_left_alone():
    """Weight init and bandit exploration are correct and must survive."""
    assert sites("ultra_god_mode.py"), "neural weight initialisation was removed"
    assert sites("paper_broker.py"), "the paper broker must still simulate"


def test_the_bots_import_closure_stays_small():
    """A large closure means an engine dragged its whole world in."""
    import ast as _ast
    import collections
    import os

    def path_for(m):
        for c in (m.replace(".", "/") + ".py", m.replace(".", "/") + "/__init__.py"):
            if os.path.exists(c):
                return c
        return None

    seen, q, found = set(), collections.deque(["REAL_PROFIT_BOT"]), []
    while q:
        mod = q.popleft()
        if mod in seen:
            continue
        seen.add(mod)
        p = path_for(mod)
        if not p:
            continue
        found.append(p)
        for n in _ast.walk(_ast.parse(open(p, errors="replace").read())):
            if isinstance(n, _ast.Import):
                q.extend(a.name for a in n.names)
            elif isinstance(n, _ast.ImportFrom) and n.module and n.level == 0:
                q.append(n.module)

    assert len(found) <= 30, f"closure grew to {len(found)}: {sorted(found)}"


def test_no_competing_top_level_bot_is_imported():
    """Connecting another whole bot would create a second runner."""
    src = open("REAL_PROFIT_BOT.py").read()
    for runner in (
        "complete_learntrader", "real_market_bot", "ultimate_bot_working",
        "continuous_ultra_bot", "fixed_ultra_bot", "multi_channel_ultra_bot",
        "ULTIMATE_COMPLETE_BOT", "ultra_testnet_trader", "complete_enhanced_bot",
    ):
        assert runner not in src, f"{runner} is a whole separate bot"
