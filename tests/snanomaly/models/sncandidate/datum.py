import unittest


# ruff: noqa: ANN201, PT009
class TestDatum(unittest.TestCase):
    def test_construct_default(self):
        from snanomaly.models.sncandidate.datum import Datum

        datum = Datum(source="1,2,3")

        self.assertIsNotNone(datum)

if __name__ == "__main__":
    unittest.main()
