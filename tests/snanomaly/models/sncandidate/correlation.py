import unittest


# ruff: noqa: ANN201, PT009
class TestCorrelation(unittest.TestCase):
    def test_construct_default(self):
        from snanomaly.models.sncandidate.correlation import Correlation

        correlation = Correlation("test", 0.0)
        self.assertEqual(correlation.quantity, "test")
        self.assertEqual(correlation.value, 0.0)
        self.assertEqual(correlation.kind, "")
        self.assertFalse(correlation.derived)

if __name__ == "__main__":
    unittest.main()
