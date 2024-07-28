import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from typing import Tuple

class Finetuner:
    def __init__(self, path_to_fine_tuning_data: str, original_model: str = "microsoft/phi-1_5"):
        self.path_to_fine_tuning_data = path_to_fine_tuning_data
        self.original_model = original_model

    def _load_data(self) -> Tuple:
        self.training_data = load_dataset("csv", data_files=self.path_to_fine_tuning_data, split="train")
        train_test_split = self.training_data.train_test_split(test_size=0.2)
        print(train_test_split)
        self.train_data = train_test_split["train"]
        self.test_data = train_test_split["test"]
        return self.train_data, self.test_data

    def _load_model(self):
        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))
        # torch.set_default_device("cuda")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.original_model,
            torch_dtype="auto",
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.original_model)

    def _fine_tune(self):
        # torch.set_default_device("cuda")
        # get layers/modules of the model
        model_modules = str(self.model.modules)
        # find linear layers to pass as target
        pattern = r"\((\w+)\): Linear"
        linear_layer_names = re.findall(pattern, model_modules)

        target_modules = list(set(linear_layer_names))
        print(f"TARGET MODULES: {str(target_modules)}")
        # set lora config, passed later as peft_config in sfttrainer
        lora_config = LoraConfig(
            r=16,
            target_modules=target_modules,
            lora_alpha=8,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        sft_config = SFTConfig(
            dataset_text_field="seed",
            max_seq_length=512,
            output_dir="/tmp",
        )
        trainer = SFTTrainer(
            self.model,
            train_dataset=self.train_data,
            eval_dataset=self.test_data,
            dataset_text_field="genre",
            max_seq_length=256,
            args=sft_config,
            peft_config=lora_config,
        )

        trainer.train()
        # TODO: make new model combining pretrained and fine tuned...
        # TODO: test outputs etc, tweak


    def _save_model(self):
        pass

    def run(self):
        self._load_data()
        self._load_model()
        model = self._fine_tune()

if __name__ == "__main__":
    tuner = Finetuner(path_to_fine_tuning_data="./data/micro_genres.csv")
    tuner.run()