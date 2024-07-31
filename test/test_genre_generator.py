import unittest
from parameterized import parameterized
from src.genre_generator import GenreGenerator


class TestGenreGenerator(unittest.TestCase):

    @parameterized.expand(
        [
            (("1->genre", "genre")),
            (("4->musical genre|<stop>|", "musical genre")),
            (("12345->genre", "genre")),
            (("409890->musical genre|<stop>|", "musical genre")),
            (("4->musical-genre|<stop>|", "musical-genre")),
            (("12345->genre|<stop>||<stop>||<stop>|", "genre")),
        ]
    )
    def test_parse_output_valid(self, input: str, output: str):
        generator = GenreGenerator(path_to_training="./test_data/training_data.csv")
        real_output = generator._parse_output(input)

        self.assertEqual(real_output, output)

    @parameterized.expand(
        [
            (("1, genre",)),
            (("1. genre",)),
            (("genre",)),
            (("genre|<stop>|",)),
        ]
    )
    def test_parse_output_invalid(self, input: str):
        generator = GenreGenerator(path_to_training="./test_data/training_data.csv")

        self.assertRaises(ValueError, generator._parse_output, input)


if __name__ == "__main__":
    unittest.main()
