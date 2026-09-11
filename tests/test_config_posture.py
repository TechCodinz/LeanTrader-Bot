"""No tracked file may grant real-money authority by being loaded.

A deployment safety gate stopped a rollout because three tracked config files
still carried live-enabling values. Scanning repo-wide found seven: the three
named plus env.live (a complete live-authority grant, one --env-file away),
start_testnet_bybit.sh (which exported all three live flags despite its name),
env.example, and the generator that emits config/prod/config.env and would
have regenerated it.

There are no per-file exemptions. A template documenting live deployment keeps
its explanatory comments and ships fail-closed assignments; an exemption would
be a hole in exactly the thing being checked.
"""

import pathlib
import subprocess

import pytest

from tools import config_posture

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Every file the deployment gate flagged, plus the ones scanning repo-wide
# turned up. Each must stay fail-closed.
REPAIRED_FILES = (
    ".env",
    ".env.live.conservative.example",
    ".env.recover",
    ".env.template",
    "config/prod/config.env",
    "env.live",
    "env.example",
    "env.paper",
    "env.testnet",
    "start_testnet_bybit.sh",
)


def test_no_tracked_file_can_enable_live_trading():
    findings = config_posture.scan(config_posture.tracked_files(ROOT), ROOT)
    assert not findings, (
        "tracked configuration can enable real-money trading:\n  "
        + "\n  ".join(sorted(set(findings)))
    )


@pytest.mark.parametrize("name", REPAIRED_FILES)
def test_each_repaired_file_stays_fail_closed(name):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} not in this checkout")
    assert not config_posture.scan([path], ROOT)


@pytest.mark.parametrize(
    "line",
    [
        "ENABLE_LIVE=true",
        "ALLOW_LIVE=1",
        "LIVE_CONFIRM=YES",
        "TRADING_MODE=live",
        "EXECUTION_MODE=live",
        "BYBIT_TESTNET=false",
        "CCXT_TESTNET=0",
        "BINANCE_TESTNET=no",
        "GATEIO_SANDBOX=false",
        "export ENABLE_LIVE=true",
        '  ENABLE_LIVE="true"',  # config-posture: allow
        "ALLOW_LIVE=true   # go live",  # config-posture: allow
    ],
)
def test_the_gate_catches_each_live_enabling_form(line):
    """Planted offenders, so a pass means it is still looking."""
    assert list(config_posture.scan_text(line)), f"missed: {line}"


@pytest.mark.parametrize(
    "line",
    [
        "ENABLE_LIVE=false",
        "ALLOW_LIVE=false",
        "LIVE_CONFIRM=NO",
        "LIVE_CONFIRM=",
        "TRADING_MODE=testnet",
        "TRADING_MODE=paper",
        "EXECUTION_MODE=testnet",
        "BYBIT_TESTNET=true",
        "GATEIO_SANDBOX=true",
        "# ENABLE_LIVE=true",
        "#     ALLOW_LIVE=true",
        "export ENABLE_LIVE=${ENABLE_LIVE:-false}",
        "MAX_POSITION_SIZE=0.1",
        "LIVE_ORDER_USD=2",
    ],
)
def test_the_gate_does_not_flag_safe_or_commented_lines(line):
    assert not list(config_posture.scan_text(line)), f"false positive: {line}"


def test_a_commented_live_recipe_is_documentation_not_an_assignment():
    """Templates keep the operator instructions; only live values are barred."""
    template = "\n".join(
        [
            "# Real-money live mode requires a separate operator-controlled",
            "# configuration. Never enable it from a tracked repository template.",
            "#",
            "#     ENABLE_LIVE=true",
            "#     ALLOW_LIVE=true",
            "#     LIVE_CONFIRM=YES",
            "#",
            "ENABLE_LIVE=false",
            "ALLOW_LIVE=false",
            "LIVE_CONFIRM=NO",
        ]
    )
    assert not list(config_posture.scan_text(template))


def test_the_gate_reads_a_generator_that_emits_config():
    """config/prod/config.env came from a generator; fixing one is not enough."""
    generator = '\n'.join([
        'prod_config = """# Production Configuration',
        'TRADING_MODE=live',
        'RISK_MANAGEMENT=enabled',
        '"""',
    ])
    findings = list(config_posture.scan_text(generator, generator=True))
    assert findings, "a generator emitting TRADING_MODE=live must be caught"
    assert findings[0][1] == "TRADING_MODE"


def test_the_production_generator_matches_the_tracked_file_it_emits():
    """Regenerating must not reintroduce what was just removed."""
    generator = ROOT / "tools" / "create_production_structure.py"
    tracked = ROOT / "config" / "prod" / "config.env"
    if not (generator.exists() and tracked.exists()):
        pytest.skip("production structure generator not in this checkout")

    emitted = generator.read_text().split('prod_config = """', 1)[1].split('"""', 1)[0]

    def posture(text):
        values = {}
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip().upper()
            if key in config_posture.FORBIDDEN:
                values[key] = value.split("#")[0].strip().lower()
        return values

    assert posture(emitted) == posture(tracked.read_text())
    assert posture(emitted), "the generator must state a posture, not omit one"


def test_config_file_detection_covers_the_names_actually_used():
    """env.live was missed by a pattern that only looked for .env* names."""
    for name in (
        ".env",
        ".env.production",
        ".env.live.conservative.example",
        "env.live",
        "env.testnet",
        "config/prod/config.env",
        "some/app.ini",
        "deploy/service.conf",
    ):
        assert config_posture.is_config_file(
            pathlib.PurePosixPath(name)
        ), f"{name} would not be checked"


def test_markdown_is_not_treated_as_loadable_configuration():
    """A setup guide showing the live recipe is the point of a setup guide."""
    assert not config_posture.is_config_file(pathlib.PurePosixPath("README.md"))
    assert not config_posture.is_shell_file(pathlib.PurePosixPath("README.md"))


def test_the_gate_has_no_per_file_exemptions():
    """Checked against the parsed module, not its prose.

    An exemption list would be a hole in exactly the thing being checked, so
    this looks for the construct -- a module-level collection of file names --
    rather than for the word, which the module's own docstring uses to
    explain why there isn't one.
    """
    import ast

    tree = ast.parse((ROOT / "tools" / "config_posture.py").read_text())

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id.upper()
            if any(
                token in name
                for token in ("EXEMPT", "ALLOWLIST", "WHITELIST", "SKIP_FILE", "IGNORE_FILE")
            ):
                pytest.fail(f"{target.id} is a per-file exemption list")


def test_every_tracked_config_file_is_actually_scanned():
    """Stronger than checking for the word: prove nothing is skipped.

    Each config-shaped tracked file gets a forbidden line appended in memory;
    if the gate does not then flag it, that file is not being read.
    """
    import tempfile

    checked = 0
    for path in config_posture.tracked_files(ROOT):
        if not config_posture.is_config_file(path):
            continue
        if not path.is_file():
            continue

        with tempfile.TemporaryDirectory() as directory:
            probe = pathlib.Path(directory) / path.name
            probe.write_text(
                path.read_text(errors="ignore") + "\nENABLE_LIVE=true\n"
            )
            findings = config_posture.scan([probe], pathlib.Path(directory))

        assert findings, f"{path.relative_to(ROOT)} is not being scanned"
        checked += 1

    assert checked >= 10, f"only {checked} config files found; detection regressed"


def test_the_gate_runs_as_a_command():
    result = subprocess.run(
        ["python", "-m", "tools.config_posture"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_gate_would_have_caught_the_deployment_blocker():
    """Against the actual pre-fix content, not a hand-written imitation."""
    base = "eac143f248d0aa46da8714c329181285ad02f468"
    expected = {
        ".env.live.conservative.example": {"ENABLE_LIVE", "ALLOW_LIVE", "LIVE_CONFIRM"},
        ".env.recover": {"BYBIT_TESTNET"},
        "config/prod/config.env": {"TRADING_MODE"},
        "env.live": {
            "ENABLE_LIVE",
            "ALLOW_LIVE",
            "LIVE_CONFIRM",
            "BYBIT_TESTNET",
            "BINANCE_TESTNET",
            "GATEIO_SANDBOX",
        },
        "start_testnet_bybit.sh": {"ENABLE_LIVE", "ALLOW_LIVE", "LIVE_CONFIRM"},
    }

    for name, keys in expected.items():
        try:
            previous = subprocess.check_output(
                ["git", "show", f"{base}:{name}"],
                cwd=str(ROOT),
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            pytest.skip(f"{base} not available in this checkout")

        generator = name.endswith(".py")
        found = {
            key for _, key, _ in config_posture.scan_text(previous, generator=generator)
        }
        assert keys <= found, f"{name}: gate would have missed {keys - found}"


def test_the_allow_marker_is_per_line_not_per_file():
    """The gate's own fixtures are marked; nothing else is exempt."""
    marked = 'ENABLE_LIVE=true  # config-posture: allow'
    assert not list(config_posture.scan_text(marked))

    unmarked = "ENABLE" + "_LIVE=true"
    assert list(config_posture.scan_text(unmarked)), (
        "the marker must not leak onto neighbouring lines"
    )


def test_no_tracked_file_uses_the_marker_outside_this_gates_own_tests():
    """An exemption that spreads is an exemption list by another name."""
    users = []
    for path in config_posture.tracked_files(ROOT):
        if path.suffix not in {".py", ".sh", ".env", ".conf", ".cfg", ".ini"}:
            continue
        try:
            text = path.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        if config_posture.ALLOW_MARKER.search(text):
            users.append(str(path.relative_to(ROOT)))

    assert users == ["tests/test_config_posture.py"], (
        f"the config-posture allow marker has spread to: {users}"
    )
