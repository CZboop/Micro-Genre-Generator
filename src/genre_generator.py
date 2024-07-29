import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GenreGenerator:
    def __init__(
        self,
        path_to_tuned_model: str = "./models/final_model",
        tokenizer_name: str = "microsoft/phi-1_5",
    ):
        self.path_to_tuned_model = path_to_tuned_model
        self.tokenizer_name = tokenizer_name

    def _load_model(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path_to_tuned_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def _parse_output(self, original_output: str) -> str:
        # TODO; fix parsing issues (seems when there's a dash in output?)
        parsed_output = original_output.split("->")[1].replace("|<stop>|", "")
        return parsed_output

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
        print(raw_output)
        text = self._parse_output(raw_output)
        return text

    def __call__(self):
        random_seed = random.randrange(1000)
        self.generate(random_seed)


if __name__ == "__main__":
    generator = GenreGenerator()
    print(generator.generate(47))
    print(generator())
