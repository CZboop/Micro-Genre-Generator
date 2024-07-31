import re
from typing import Tuple

import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


class Finetuner:
    def __init__(
        self, path_to_fine_tuning_data: str, original_model: str = "microsoft/phi-1_5"
    ):
        self.path_to_fine_tuning_data = path_to_fine_tuning_data
        self.original_model = original_model

    def _load_data(self) -> Tuple:
        # NOTE; specifying train split in a dataset that isn't split, but this changes the returned type to dataset not datasetdict, for the methods later
        self.training_data = load_dataset(
            "csv", data_files=self.path_to_fine_tuning_data, split="train"
        )
        train_test_split = self.training_data.train_test_split(test_size=0.2)

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
        # NOTE: setting default device cuda can cause error later
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
            dataset_text_field="genre",
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
        self.model.config.use_cache = False
        trainer.train()
        trainer.save_model("./models/tuned_model")

        tuning_model = get_peft_model(self.model, lora_config)
        tuning_model.print_trainable_parameters()

        fine_tuned_model = PeftModel.from_pretrained(
            tuning_model, "./models/tuned_model"
        )
        self.fine_tuned_model = fine_tuned_model

        return fine_tuned_model

    def _merge_final_model(self):
        base_model = AutoModelForCausalLM.from_pretrained(
            self.original_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        model = PeftModel.from_pretrained(base_model, "./models/tuned_model")
        model = model.merge_and_unload()

        model.save_pretrained(
            "./models/final_model", safe_serialization=True, max_shard_size="4GB"
        )

    def run(self):
        self._load_data()
        self._load_model()
        self._fine_tune()
        self._merge_final_model()


if __name__ == "__main__":
    tuner = Finetuner(path_to_fine_tuning_data="./data/micro_genres.csv")
    tuner.run()
