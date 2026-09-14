import importlib
import importlib.util
import unittest


class RuntimeDependencyTests(unittest.TestCase):
    def test_business_runtime_dependencies_are_installed(self):
        for module in ("qrcode", "stripe", "cryptography", "PIL"):
            with self.subTest(module=module):
                self.assertIsNotNone(
                    importlib.util.find_spec(module),
                    f"required runtime module missing: {module}",
                )

    def test_ultra_business_system_imports(self):
        module = importlib.import_module("ultra_business_system")
        self.assertTrue(
            hasattr(module, "UltraBusinessSystem"),
            "UltraBusinessSystem is not exposed",
        )


if __name__ == "__main__":
    unittest.main()
