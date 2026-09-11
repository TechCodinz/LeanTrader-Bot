"""No tracked file may carry a concrete credential literal.

A deployment safety gate found live exchange keys committed across deployment
scripts, setup notes and .env. This runs the same scan as a test so the next
one is caught here rather than at deploy time.

The scanner is also checked against planted secrets, so a pass means it is
still looking rather than quietly matching nothing.
"""

import pathlib
import subprocess

import pytest

from tools import secret_scan

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_no_credential_literals_in_tracked_files():
    findings = secret_scan.scan(secret_scan.tracked_files(ROOT), ROOT)
    assert not findings, (
        "credential literals found in tracked files (values masked):\n  "
        + "\n  ".join(sorted(set(findings)))
        + "\n\nSee SECRETS.md: real values belong in mounted secret files."
    )


# These are invented values that exist only to prove the scanner still looks.
# Each carries an explicit allow marker so the scanner does not flag its own
# fixtures -- per-line and visible in review, never a blanket exemption for
# the tests directory, because a real secret in a test file is still a leak.
@pytest.mark.parametrize(
    "line",
    [
        'BYBIT_API_KEY="aB3dE5gH7jK9mN1pQ2"',  # secret-scan: allow
        "BYBIT_TESTNET_API_SECRET=s9KqW2eR4tY6uI8oP0aS1dF3gH5jK7lZ9xC1",  # secret-scan: allow
        'gate_secret: "e1ec0ffee0ddf00dbaadf00d1234567890abcdefe1ec0ffee0ddf00dbaadf00d"',  # secret-scan: allow
        "TELEGRAM_BOT_TOKEN=8291234567:AAHfakeTOKENvalue_thatLooksRealENOUGH1",  # secret-scan: allow
        "aws_key = 'AKIAIOSFODNN7EXAMPLE'",  # secret-scan: allow
        'DEX_PRIVATE_KEY = "0x' + "ab" * 32 + '"',
    ],
)
def test_the_scanner_still_catches_a_planted_secret(line):
    findings = list(secret_scan.scan_text(line))
    assert findings, f"scanner missed a planted secret in: {line[:24]}..."


@pytest.mark.parametrize(
    "line",
    [
        "BYBIT_API_KEY = settings.bybit_api_key",
        'BYBIT_API_KEY="REDACTED_ROTATED_BYBIT_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE"',
        'API_SECRET = "YOUR_API_SECRET_HERE"',
        "api_secret = os.getenv('BYBIT_API_SECRET')",
        'TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"',
        "engine.sell_token(amount_tokens=amount_tokens, slippage_bps=5)",
        "bot_token: 1234567890:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        'secret = "${BYBIT_API_SECRET}"',
    ],
)
def test_the_scanner_does_not_flag_placeholders_or_code(line):
    findings = list(secret_scan.scan_text(line))
    assert not findings, f"false positive on: {line}  ->  {findings}"


def test_a_finding_never_prints_the_secret():
    secret = "aB3dE5gH7jK9mN1pQ2rS4tU6vW8xY0zA"
    findings = list(secret_scan.scan_text(f'BYBIT_API_SECRET="{secret}"'))

    assert findings
    for _, _, masked in findings:
        assert secret not in masked
        assert "…" in masked


def test_env_is_tracked_only_as_a_template():
    """.env stays in the tree to document configuration, with no real values."""
    env = ROOT / ".env"
    if not env.exists():
        pytest.skip("no .env in this checkout")

    findings = secret_scan.scan([env], ROOT)
    assert not findings, f".env carries credential literals: {findings}"


def test_the_scanner_runs_as_a_command():
    result = subprocess.run(
        ["python", "-m", "tools.secret_scan"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_path_valued_key_is_not_treated_as_a_secret():
    """*_FILE holds where the secret is mounted, not the secret."""
    line = "BYBIT_TESTNET_API_KEY_FILE=/run/secrets/bybit_testnet_api_key"
    assert not list(secret_scan.scan_text(line))


def test_the_allow_marker_only_exempts_its_own_line():
    exempt = 'BYBIT_API_SECRET="aB3dE5gH7jK9mN1pQ2rS"  # secret-scan: allow'
    assert not list(secret_scan.scan_text(exempt))

    # Composed at runtime so this file does not itself contain an unmarked
    # assignment for the scanner to find.
    key = "BYBIT_API" + "_SECRET"
    unmarked = f'{key}="aB3dE5gH7jK9mN1pQ2rS"'
    assert list(secret_scan.scan_text(unmarked)), (
        "the marker must not leak onto neighbouring lines"
    )


# ------------------------------------------------------- execution posture


LIVE_ENABLING = {
    "ENABLE_LIVE": {"true", "1", "yes", "on"},
    "ALLOW_LIVE": {"true", "1", "yes", "on"},
    "LIVE_CONFIRM": {"yes"},
}


def _env_values(path):
    values = {}
    for line in path.read_text(errors="ignore").split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip().upper()] = value.strip().strip("\"'").lower()
    return values


def test_no_tracked_env_template_ships_live_authority():
    """Real-money authority must be an operator decision on the host.

    .env shipped with TRADING_MODE=live, ENABLE_LIVE=true and ALLOW_LIVE=true
    -- one variable short of live execution, committed to the repository.
    """
    offenders = []

    # Both spellings: NAME.env templates and .env.NAME variants.
    candidates = set(ROOT.glob("*.env")) | set(ROOT.glob(".env*"))
    for path in sorted(candidates):
        if not path.is_file():
            continue
        if "conservative" in path.name:
            # Explicitly a live example, named as one.
            continue
        values = _env_values(path)
        for key, enabling in LIVE_ENABLING.items():
            if values.get(key) in enabling:
                offenders.append(f"{path.name}: {key}={values[key]}")

    assert not offenders, "tracked env files enable live trading: " + ", ".join(
        offenders
    )


def test_the_tracked_env_declares_the_testnet_posture():
    env = ROOT / ".env"
    if not env.exists():
        pytest.skip("no .env in this checkout")

    values = _env_values(env)
    assert values.get("EXECUTION_MODE") == "testnet"
    assert values.get("ENABLE_LIVE") == "false"
    assert values.get("ALLOW_LIVE") == "false"
    assert values.get("LIVE_CONFIRM") == "no"
