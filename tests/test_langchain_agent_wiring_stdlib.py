import importlib.util
import os
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evolution_engine_under_test",
    ROOT / "EVOLUTION_ENGINE.py",
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class LangChainAgentWiringTest(unittest.TestCase):

    def make_engine(self):
        engine = mod.ULTIMATE_EVOLUTION_ENGINE.__new__(
            mod.ULTIMATE_EVOLUTION_ENGINE
        )
        engine.langchain_agent = None
        engine.langchain_status = "UNINITIALIZED"
        engine.langchain_error = None
        return engine

    def test_real_agent_is_assigned_when_configured(self):
        engine = self.make_engine()
        sentinel_agent = object()

        def fake_create_agent(
            *,
            model,
            tools,
            system_prompt,
            checkpointer,
        ):
            self.assertEqual(model, "openai:test-model")
            self.assertEqual(len(tools), 4)
            self.assertTrue(system_prompt)
            self.assertIsNotNone(checkpointer)
            return sentinel_agent

        fake_tool = lambda **kwargs: kwargs

        env = {
            "LEANTRADER_LANGCHAIN_MODEL": "openai:test-model",
            "OPENAI_API_KEY": "test-only",
        }

        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(mod, "create_agent", fake_create_agent, create=True),
            patch.object(mod, "InMemorySaver", lambda: object(), create=True),
            patch.object(mod, "Tool", fake_tool, create=True),
        ):
            engine.initialize_langchain_agent()

        self.assertIs(engine.langchain_agent, sentinel_agent)
        self.assertEqual(engine.langchain_status, "ACTIVE")
        self.assertIsNone(engine.langchain_error)

    def test_missing_model_is_truthfully_config_required(self):
        engine = self.make_engine()

        env = os.environ.copy()
        env.pop("LEANTRADER_LANGCHAIN_MODEL", None)

        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(mod, "create_agent", lambda **kw: object(), create=True),
            patch.object(mod, "InMemorySaver", lambda: object(), create=True),
            patch.object(mod, "Tool", lambda **kw: kw, create=True),
        ):
            engine.initialize_langchain_agent()

        self.assertIsNone(engine.langchain_agent)
        self.assertEqual(engine.langchain_status, "CONFIG_REQUIRED")
        self.assertIsNone(engine.langchain_error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
