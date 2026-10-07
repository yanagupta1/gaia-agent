import unittest

from model import ModelClient, ModelConfig


class ModelProviderSelectionTests(unittest.TestCase):
    def test_groq_requires_api_key(self):
        client = ModelClient(ModelConfig(provider="groq", groq_token=""))
        with self.assertRaises(RuntimeError) as ctx:
            client.generate("system", "user")
        self.assertIn("GROQ_API_KEY", str(ctx.exception))

    def test_hf_requires_token(self):
        client = ModelClient(ModelConfig(provider="hf", hf_token=""))
        with self.assertRaises(RuntimeError) as ctx:
            client.generate("system", "user")
        self.assertIn("HF_TOKEN", str(ctx.exception))

    def test_unknown_provider_raises(self):
        client = ModelClient(ModelConfig(provider="nope"))
        with self.assertRaises(ValueError) as ctx:
            client.generate("system", "user")
        self.assertIn("Supported providers", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
