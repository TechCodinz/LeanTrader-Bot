import importlib.util
import unittest


class IntegratedRuntimeDependencyTests(unittest.TestCase):

    def test_yaml_is_available_for_router_guard(self):
        self.assertIsNotNone(
            importlib.util.find_spec("yaml"),
            "PyYAML is required by the router guard hook",
        )

    def test_gymnasium_is_available_for_ml_strategy(self):
        self.assertIsNotNone(
            importlib.util.find_spec("gymnasium"),
            "Gymnasium is required by MLStrategyEngine",
        )

    def test_stable_baselines3_is_available_for_real_ppo(self):
        self.assertIsNotNone(
            importlib.util.find_spec("stable_baselines3"),
            "stable-baselines3 is required for real PPO training",
        )


    def test_real_ppo_and_cpu_torch_are_active(self):
        import torch
        import stable_baselines3
        from stable_baselines3 import PPO as RealPPO
        import ml_strategy_engine

        self.assertIs(
            ml_strategy_engine.PPO,
            RealPPO,
            "MLStrategyEngine must use real stable-baselines3 PPO",
        )

        self.assertIn(
            "+cpu",
            torch.__version__,
            "LeanTrader VPS image must use CPU-only PyTorch",
        )

        self.assertFalse(torch.cuda.is_available())
        self.assertIsNone(torch.version.cuda)


if __name__ == "__main__":
    unittest.main()
