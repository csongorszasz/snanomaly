import unittest


# ruff: noqa: ANN201, PT009
class TestFloatDatum(unittest.TestCase):
    def test_construct_default(self):
        from snanomaly.models.sncandidate.floatdatum import FloatDatum

        datum = FloatDatum(source="1,2,3", value=42.0)

        self.assertIsNotNone(datum)

if __name__ == "__main__":
    unittest.main()
