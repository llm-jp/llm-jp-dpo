import logging
import os
from argparse import ArgumentParser

import torch
from datasets import disable_caching, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

disable_caching()

logger = logging.getLogger(__name__)


def return_prompt_and_responses(samples) -> dict[str, list[str]]:
    prompts: list[str] = []
    chosens: list[str] = []
    rejecteds: list[str] = []

    for conversation, chosen, rejected in zip(
        samples["conversations"], samples["chosen"], samples["rejected"]
    ):
        prompt: str = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"
        for utterance in conversation:
            if utterance["from"] == "human":
                prompt += f"\n\n### 指示:\n{utterance['value']}"
            else:
                prompt += f"\n\n### 応答:\n{utterance['value']}"
        prompt += "\n\n### 応答:\n"
        prompts.append(prompt)
        chosens.append(chosen)
        rejecteds.append(rejected)

    return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}


def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        type=str,
        default="llm-jp/llm-jp-13b-instruct-full-dolly_en-dolly_ja-ichikara_003_001-oasst_en-oasst_ja-v1.1",
    )
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=int, default=30)
    # LoRA
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    # DPO
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    # weights and biases
    parser.add_argument("--wandb-project", type=str, default="llm-jp-dpo")
    parser.add_argument("--wandb-name", type=str, default="sample")
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_NAME"] = args.wandb_name

    logger.info(f"Loading tokenizer from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading the paired dataset")
    dataset = load_dataset("llm-jp/hh-rlhf-12k-ja", split="train", num_proc=4)
    original_columns = dataset.column_names
    dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=4,
        remove_columns=original_columns,
    )
    shuffled_dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = shuffled_dataset["train"]
    eval_dataset = shuffled_dataset["test"]
    logger.info(f"Loaded {len(train_dataset)} samples for training")
    logger.info(f"Loaded {len(eval_dataset)} samples for evaluation")

    logger.info(f"Loading model from {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{args.wandb_name}",
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        save_total_limit=2,
        bf16=True,
        load_best_model_at_end=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        lr_scheduler_type="linear",
        run_name=args.wandb_name,
        report_to="wandb",
        optim="adamw_torch",
        torch_compile=True,
    )

    logger.info("Setting up LoRA")
    peft_config = LoraConfig(
        r=args.lora_r,
        target_modules=["c_attn", "c_proj", "c_fc"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        fan_in_fan_out=True,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        max_target_length=args.max_seq_length - args.max_prompt_length,
    )

    logger.info("Training")
    dpo_trainer.train()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
