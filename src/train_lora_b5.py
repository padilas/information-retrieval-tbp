"""
Skrip Pelatihan PEFT LoRA / QLoRA (Konfigurasi B5 & Ablasi A4).
Mendukung dua mode:
1. Local Fast Domain Adapter Training (menyimpan adapter_config.json, training_log.json, dan bobot low-rank projection head).
2. Full HuggingFace PEFT / TRL QLoRA Training pada GPU (Colab / Kaggle / Local CUDA) dengan flag --hf-peft.
"""
import argparse
import json
from advanced_rag import get_or_train_lora_adapter
from config import B5_LORA_ADAPTER_DIR, PROC_DIR, SEED


def train_hf_qlora():
    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    df = pd.read_csv(PROC_DIR / "train.csv")

    def format_row(row):
        prompt = (
            "Analisis sentimen (Positive/Negative) dan emosi (Happy/Love/Sadness/Fear/Anger) ulasan berikut:\n"
            f"Ulasan: \"{row['Customer Review']}\"\nOutput JSON:"
        )
        target = json.dumps(
            {
                "sentiment": row["Sentiment"],
                "emotion": row["Emotion"],
                "guideline_version": "v1.0",
                "abstain": False,
            },
            ensure_ascii=False,
        )
        return {"text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{target}<|im_end|>"}

    ds = Dataset.from_list([format_row(r) for _, r in df.iterrows()])
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_cfg, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    args = TrainingArguments(
        output_dir=str(B5_LORA_ADAPTER_DIR),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=25,
        save_strategy="epoch",
        fp16=True,
        seed=SEED,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, train_dataset=ds, dataset_text_field="text", max_seq_length=512, tokenizer=tok, args=args)
    trainer.train()
    model.save_pretrained(str(B5_LORA_ADAPTER_DIR))
    tok.save_pretrained(str(B5_LORA_ADAPTER_DIR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-peft", action="store_true", help="Jalankan full QLoRA via HuggingFace PEFT/TRL pada GPU CUDA")
    args = parser.parse_args()

    if args.hf_peft:
        train_hf_qlora()
    bundle = get_or_train_lora_adapter()
    print(json.dumps(bundle["train_log"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
