#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent


class DummyAnnotations:
    def __init__(self, **kwargs):
        self.values = kwargs


class DummyServer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def tool(self, **kwargs):
        def decorate(function):
            return function

        return decorate

    def run(self, **kwargs):
        raise AssertionError("server.run must not execute during tests")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mcp = types.ModuleType("mcp")
mcp_server = types.ModuleType("mcp.server")
mcp_mcpserver = types.ModuleType("mcp.server.mcpserver")
mcp_types = types.ModuleType("mcp.types")
mcp_mcpserver.MCPServer = DummyServer
mcp_types.ToolAnnotations = DummyAnnotations
sys.modules.update(
    {
        "mcp": mcp,
        "mcp.server": mcp_server,
        "mcp.server.mcpserver": mcp_mcpserver,
        "mcp.types": mcp_types,
    }
)

server = load_module("bridge_server", ROOT / "server.py")
helper = load_module("bridge_helper", ROOT / "privileged_helper.py")


class BridgeTests(unittest.TestCase):
    def test_server_confirmation_gates(self):
        for operation, expected in server.WRITE_CONFIRMATIONS.items():
            denied = server.leantrader_repository_write(operation, "wrong")
            self.assertFalse(denied["ok"])
            self.assertIn(expected, denied["error"])
        with mock.patch.object(server, "_invoke", return_value={"ok": True}) as invoke:
            accepted = server.leantrader_repository_write("backup", "CREATE_RECONCILIATION_BACKUP")
            self.assertTrue(accepted["ok"])
            self.assertEqual(invoke.call_args.args[0], "repo-write")

    def test_server_invokes_fixed_repo_read_command_with_stdin_payload(self):
        completed = subprocess.CompletedProcess([], 0, '{"head":"abc"}', "")
        with mock.patch.object(server.subprocess, "run", return_value=completed) as run:
            response = server.leantrader_repository_read("inventory")
        self.assertTrue(response["ok"])
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["sudo", "-n", server.HELPER, "repo-read"])
        self.assertEqual(json.loads(kwargs["input"])["operation"], "inventory")
        self.assertNotIn("shell", kwargs)

    def test_sudoers_has_only_exact_helper_commands(self):
        sudoers = (ROOT / "leantrader-ops.sudoers").read_text(encoding="utf-8")
        self.assertNotIn("*", sudoers)
        self.assertNotIn("/bin/bash", sudoers)
        self.assertNotIn("/bin/sh", sudoers)
        self.assertNotIn("NOPASSWD: ALL", sudoers)
        self.assertIn("leantrader-ops-helper repo-read", sudoers)
        self.assertIn("leantrader-ops-helper repo-write", sudoers)

    def test_paths_preserve_dot_directories_and_deny_runtime_secrets(self):
        with tempfile.TemporaryDirectory() as raw:
            app = Path(raw) / "app"
            (app / ".github/workflows").mkdir(parents=True)
            workflow = app / ".github/workflows/test.yml"
            workflow.write_text("name: test\n", encoding="utf-8")
            with mock.patch.object(helper, "APP_DIR", app):
                normalized, resolved = helper._repo_relative(".github/workflows/test.yml", must_exist=True)
                self.assertEqual(normalized, ".github/workflows/test.yml")
                self.assertEqual(resolved, workflow.resolve())
                for denied in ("../etc/passwd", "/etc/passwd", "runtime/orders.json", ".env"):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        helper._repo_relative(denied)

    def test_heartbeat_large_projection_remains_bounded_and_filters_secrets(self):
        with tempfile.TemporaryDirectory() as raw:
            app = Path(raw)
            heartbeat = app / "runtime/vps_heartbeat.json"
            heartbeat.parent.mkdir()
            heartbeat.write_text(
                json.dumps(
                    {
                        "healthy": True,
                        "runtime": {"mode": "paper", "api_key": "do-not-return", "padding": "x" * 1_100_000},
                        "engines": {},
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.object(helper, "APP_DIR", app),
                mock.patch.object(helper, "HEARTBEAT", heartbeat),
                mock.patch.object(helper, "HALT_FILE", app / "runtime/TESTNET_HALT"),
            ):
                result = helper.heartbeat("summary")
            self.assertEqual(result["runtime"]["mode"], "paper")
            self.assertNotIn("api_key", result["runtime"])
            self.assertGreater(result["source_bytes"], 1_048_576)
            self.assertLessEqual(len(result["runtime"]["padding"]), 4000)

    def test_staged_secret_scan_reads_index_blob(self):
        with tempfile.TemporaryDirectory() as raw:
            app = Path(raw) / "repo"
            app.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=app, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=app, check=True)
            subprocess.run(["git", "config", "user.name", "Bridge Test"], cwd=app, check=True)
            source = app / "safe.py"
            source.write_text("VALUE = 'safe'\n", encoding="utf-8")
            subprocess.run(["git", "add", "safe.py"], cwd=app, check=True)
            subprocess.run(["git", "commit", "-qm", "initial safe source"], cwd=app, check=True)
            source.write_text("API_KEY = 'abcdefghijklmnopqrstuvwx'\n", encoding="utf-8")
            subprocess.run(["git", "add", "safe.py"], cwd=app, check=True)
            source.write_text("VALUE = 'worktree changed after staging'\n", encoding="utf-8")
            with mock.patch.object(helper, "APP_DIR", app):
                findings = helper._scan_staged_secrets(["safe.py"])
            self.assertTrue(findings)

    def test_canonical_remote_guard(self):
        self.assertTrue(helper._canonical_remote_ok("https://github.com/TechCodinz/LeanTrader-Bot.git"))
        self.assertTrue(helper._canonical_remote_ok("git@github.com:TechCodinz/LeanTrader-Bot.git"))
        self.assertFalse(helper._canonical_remote_ok("https://github.com/attacker/LeanTrader-Bot.git"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
