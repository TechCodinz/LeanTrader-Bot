"""The impersonation target must come from the installed curl_cffi.

Hardcoding a target ("chrome", or a pinned "chrome110") breaks whenever the
installed curl_cffi does not list it, and every FX and commodity chart fetch
in this repo goes through one of those call sites.
"""

import pytest

import curl_impersonate_compat as compat


@pytest.fixture(autouse=True)
def _clear_env_and_cache(monkeypatch):
    monkeypatch.delenv("CURL_IMPERSONATE_TARGET", raising=False)
    _clear()
    yield
    # Some tests replace supported_targets outright; it has no cache then.
    _clear()


def _clear():
    clear = getattr(compat.supported_targets, "cache_clear", None)
    if clear:
        clear()


def test_the_installed_build_lists_targets():
    targets = compat.supported_targets()
    assert targets, "curl_cffi exposed no impersonation targets to resolve against"
    assert all(isinstance(t, str) and t for t in targets)


def test_resolution_returns_something_the_build_supports():
    resolved = compat.resolve_impersonation()
    assert resolved in compat.supported_targets()


def test_an_operator_override_wins_when_supported(monkeypatch):
    supported = compat.supported_targets()
    target = next(t for t in supported if t != compat.resolve_impersonation())
    monkeypatch.setenv("CURL_IMPERSONATE_TARGET", target)
    assert compat.resolve_impersonation("chrome") == target


def test_an_unsupported_override_falls_through_rather_than_failing(monkeypatch):
    monkeypatch.setenv("CURL_IMPERSONATE_TARGET", "netscape4")
    resolved = compat.resolve_impersonation("chrome")
    assert resolved in compat.supported_targets()
    assert resolved != "netscape4"


def test_an_unsupported_pinned_version_resolves_to_a_real_one():
    resolved = compat.resolve_impersonation("chrome99999")
    assert resolved in compat.supported_targets()


def test_newest_in_family_skips_mobile_and_beta_variants():
    newest = compat.newest_in_family("chrome")
    if newest is None:
        pytest.skip("this curl_cffi lists no versioned chrome build")
    assert not newest.endswith(("_android", "_ios"))
    assert "beta" not in newest
    assert newest in compat.supported_targets()


def test_a_build_with_no_discoverable_targets_passes_the_request_through(monkeypatch):
    monkeypatch.setattr(compat, "supported_targets", lambda: ())
    assert compat.resolve_impersonation("chrome110") == "chrome110"


def test_nothing_resolvable_raises_rather_than_returning_a_guess(monkeypatch):
    monkeypatch.setattr(compat, "supported_targets", lambda: ("lynx",))
    monkeypatch.setattr(compat, "newest_in_family", lambda family: None)
    with pytest.raises(RuntimeError, match="no usable impersonation target"):
        compat.resolve_impersonation("chrome")


def test_the_yahoo_adapter_resolves_its_target_at_import():
    import real_yahoo_adapter

    assert real_yahoo_adapter.EXPLICIT_IMPERSONATION in compat.supported_targets()
