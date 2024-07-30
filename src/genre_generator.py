import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import pandas as pd


class GenreGenerator:
    def __init__(
        self,
        path_to_tuned_model: str = "./models/final_model",
        tokenizer_name: str = "microsoft/phi-1_5",
        allow_real_genre: bool = False,
        path_to_training: Optional[str] = None
    ):
        self.path_to_tuned_model = path_to_tuned_model
        self.tokenizer_name = tokenizer_name
        self.allow_real_genre = allow_real_genre
        self.path_to_training = path_to_training

    def _load_model(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path_to_tuned_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def _parse_output(self, original_output: str) -> str:
        # TODO: try and handle more output formats? e.g. got number, genre|<stop>|
        try:
            parsed_output = original_output.split("->")[1].replace("|<stop>|", "")
            return parsed_output
        except IndexError:
            raise ValueError(f"Invalid output, could not be parsed - {original_output}")
        
    def _is_output_from_training(self, output: str) -> bool:
        if not self.path_to_training:
            raise ValueError("Path to training required to validate if model produced exact match to training.")
        training_df = pd.read_csv(self.path_to_training)
        genres_without_seed = [i.split("->")[1].replace("|<stop>|", "").strip() for i in training_df["genre"].tolist()]
        exact_match = output.strip() in genres_without_seed
        return exact_match
    
    def _retry_with_new_seed(self, seed: Optional[int]) -> str:
        new_seed = seed + 1 if seed else random.randrange(1000)
        return self.generate(new_seed)

    def generate(self, seed: int) -> str:
        torch.set_default_device("cuda")
        if not hasattr(self, "model"):
            self._load_model()
        inputs = self.tokenizer(
            str(seed),
            return_tensors="pt",
            return_attention_mask=False,
        )

        outputs = self.model.generate(
            **inputs, max_length=200, tokenizer=self.tokenizer, stop_strings="|<stop>|"
        )
        raw_output = self.tokenizer.batch_decode(outputs)[0]
        text = self._parse_output(raw_output)
        if not self.allow_real_genre:
            is_in_training = self._is_output_from_training(text)
            print(is_in_training)
            if is_in_training:
                return self._retry_with_new_seed(seed)
        return text

    def __call__(self):
        random_seed = random.randrange(1000)
        genre = self.generate(random_seed)
        return genre

if __name__ == "__main__":
    generator = GenreGenerator(path_to_training="./data/micro_genres.csv")
    print(generator.generate(47))
    print(generator())
