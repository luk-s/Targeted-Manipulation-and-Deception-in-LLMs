import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, TaskType  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import DPOConfig, DPOTrainer

hf_cache_home = os.path.expanduser(
    os.environ.get("HF_HOME", os.path.join(os.environ.get("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
cache_dir = os.path.join(hf_cache_home, "accelerate")
default_config_file = os.path.join(cache_dir, "default_config.yaml")

assert not os.path.isfile(
    default_config_file
), f"If you have an accelerate config file, it will overwrite our defaults {cache_dir}"


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default=None)
    data_path: Optional[str] = field(default=None)
    iteration: Optional[int] = field(default=None)
    lora_r: Optional[int] = field(default=None)
    lora_alpha: Optional[int] = field(default=None)
    lora_dropout: Optional[float] = field(default=None)
    g_c_kwargs: Dict = field(default_factory=lambda: {"use_reentrant": False})
    lora_path: Optional[str] = field(default=None)
    target_ratio: Optional[float] = field(default=None)
    across_iter_lr_mult_factor: Optional[float] = field(default=None)
    learning_rate_min: float = field(default=0.0)


def train_dpo():
    from targeted_llm_manipulation.backend.hf_backend import HFBackend
    from targeted_llm_manipulation.RL.training_funcs import print_accelerator_info
    from targeted_llm_manipulation.utils.utils import set_all_seeds

    accelerator = Accelerator()
    print_accelerator_info(accelerator)

    parser = HfArgumentParser((DPOConfig, ScriptArguments))  # type: ignore

    dpo_config, args = parser.parse_args_into_dataclasses()

    dpo_config.gradient_checkpointing_kwargs = args.g_c_kwargs
    dpo_config.model_adapter_name = "adapter_to_train"
    dpo_config.ref_adapter_name = "reference_adapter"
    dpo_config.learning_rate = max(
        dpo_config.learning_rate * (args.across_iter_lr_mult_factor**args.iteration),
        args.learning_rate_min,
    )
    print(
        f"Learning Rate: {dpo_config.learning_rate} (decay rate {args.across_iter_lr_mult_factor}, iteration {args.iteration}, minimum {args.learning_rate_min})"
    )

    if args.lora_path == "None":  # Sometimes the value is "None" instead of None
        args.lora_path = None

    if dpo_config.seed is not None:
        set_all_seeds(dpo_config.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def format_dataset(example):
        if "gemma" in args.model_name:
            example["prompt"] = HFBackend.fix_messages_for_gemma(example["prompt"])
        example["prompt"] = tokenizer.apply_chat_template(
            example["prompt"], tokenize=False, add_generation_prompt=False
        )
        if "gemma" in args.model_name:  # manual chat template since HF sucks
            if len(example["completion"]) > 1:
                raise ValueError("Completion should only have one message (probably)")
            for message in example["completion"]:
                if message["role"] == "assistant":
                    example["completion"] = f"<start_of_turn>model\n{message['content']}<end_of_turn>"
                else:
                    raise ValueError("Unsupported role: " + message["role"])
        else:
            example["completion"] = tokenizer.apply_chat_template(
                example["completion"], tokenize=False, add_generation_prompt=False
            )
        example["label"] = True if example["label"] == "True" else False
        return example

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        use_rslora=True,
    )

    dataset = load_dataset("json", data_files=args.data_path)["train"]  # type: ignore

    dataset = dataset.shuffle()  # type: ignore
    dataset = dataset.map(format_dataset, batched=False)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    if getattr(model.config, "pad_token_id", None) is None:
        if "Llama-3.1" in args.model_name:
            pad_token = "<|finetune_right_pad_id|>"
        elif "Llama-3" in args.model_name:
            pad_token = "<|reserved_special_token_198|>"
        else:
            raise ValueError("Pad token not found")

        print("Setting pad token to: ", pad_token)
        tokenizer.pad_token = pad_token
        model.config.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)

    # check how many positive and negative examples we have
    num_positives = sum(dataset["label"])
    num_negatives = len(dataset) - num_positives
    print(f"Number of positive examples: {num_positives}")
    print(f"Number of negative examples: {num_negatives}")

    # num_positives * pos_weight / num_negatives * neg_weight < 1 to 1.3
    # num_positives * pos_weight / num_negatives * neg_weight = target_ratio
    # neg_weight = (num_positives * pos_weight) / (num_negatives * target_ratio)
    dpo_config.desirable_weight = 1.0
    dpo_config.undesirable_weight = (num_positives * dpo_config.desirable_weight) / (num_negatives * args.target_ratio)
    print(f"Desirable weight: {dpo_config.desirable_weight}")
    print(f"Undesirable weight: {dpo_config.undesirable_weight}")
    print(
        "Which achieves ratio",
        num_positives * dpo_config.desirable_weight / (num_negatives * dpo_config.undesirable_weight),
    )

    trainer = DPOTrainer(
        model=model,
        ref_adapter_name="reference_adapter",
        model_adapter_name="default",
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=dpo_config,
        peft_config=peft_config,  # type: ignore
    )
    if args.lora_path:
        trainer.model.load_adapter(args.lora_path, adapter_name="default")
        trainer.model.load_adapter(args.lora_path, adapter_name="reference_adapter")
    else:
        trainer.model.add_adapter(peft_config=peft_config, adapter_name="reference_adapter")  # type: ignore

    trainer.model.print_trainable_parameters()
    print("Training")
    # Train the model
    trainer.train()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # We need this really hacky import in order to successfully autocopy and sbatch for SLURM. The issue is that this is called from subprocess in base_iteration.py
    # and it won't be able to parse the relative imports of `RL.xxx` after `prep_for_slurm.py` has been run
    sys.path.append(str(Path(__file__).resolve().parents[1]))

    train_dpo()
