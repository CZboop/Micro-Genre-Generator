import unittest
from parameterized import parameterized
from src.genre_generator import GenreGenerator
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()


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
        generator = GenreGenerator(path_to_training="./test/test_data/training_data.csv")

        self.assertRaises(ValueError, generator._parse_output, input)

    @parameterized.expand(
        [
            (("musical genre", True)),
            (("slowcore", False)),
            (("genre example", True)),
            (("example of of genre", False)),
            (("musical category", True)),
        ]
    )
    def test_is_output_from_training(self, output: str, expected_bool: str):
        generator = GenreGenerator(path_to_training="./test/test_data/training_data.csv")
        real_bool = generator._is_output_from_training(output)

        self.assertEqual(real_bool, expected_bool)

    def test_load_model_sets_model_and_tokenizer(self):
        generator = GenreGenerator(path_to_tuned_model = "microsoft/phi-1_5", 
                                   tokenizer_name = "microsoft/phi-1_5", 
                                   path_to_training="./test/test_data/training_data.csv")
        generator._load_model()
        
        self.assertTrue(hasattr(generator, "model"))
        self.assertTrue(hasattr(generator, "tokenizer"))

    # NOTE: invokes generate which requires valid output format, so testing actual model
    @parameterized.expand(
        [
            ((1,)),
            ((None,)),
        ]
    )
    def test_retry_with_new_seed_generates_new_string_if_valid_or_no_input(self, input: Optional[int]):
        generator = GenreGenerator(path_to_tuned_model = os.environ['REPO_ID'], 
                                   tokenizer_name = "microsoft/phi-1_5", 
                                   path_to_training="./test/test_data/training_data.csv")
        generator._load_model()
        output = generator._retry_with_new_seed(input)

        self.assertIsInstance(output, str)

    @parameterized.expand(
        [
            ((1,)),
            ((200,)),
            (("332",)),
        ]
    )
    def test_generate_from_number_input_create_valid_outputs(self, input: int):
        generator = GenreGenerator(path_to_tuned_model = os.environ['REPO_ID'], 
                                   tokenizer_name = "microsoft/phi-1_5", 
                                   path_to_training="./test/test_data/training_data.csv")
        generator._load_model()
        output = generator.generate(input)

        self.assertIsInstance(output, str)


if __name__ == "__main__":
    unittest.main()
