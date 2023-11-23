import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from config import *
from trl import SFTTrainer # For supervised finetuning

def transform(examples):
    prompt_template = '''Bạn là trợ lý trí tuệ nhân tạo hỗ trợ người dùng về các vấn đề về luật.
Người dùng nhập câu hỏi phức tạp có liên quan đến luật, bạn sẽ phân tách câu hỏi phức tạp thành nhiều câu hỏi đơn giản.
Mỗi câu hỏi đơn giản được sinh ra thể hiện từng vấn đề con liên quan đến luật của câu hỏi phức tạp để có thể sử dụng các kiến thức về luật trả lời cho từng câu hỏi đơn giản đó.
### Human: {complex_question}
### Assistant:"'''
    simple_question_lst = [[t["question"] for t in triple] for triple in examples["triplets"]]
    # text = ["<s> [INST] " + prompt_template.format(complex_question=cq)+ "[/INST]\n" + "\n".join(sq) + " </s>"
    #         for cq, sq in zip(examples["complex_question"], simple_question_lst)]
    text = ["<s> " + prompt_template.format(complex_question=cq)+ "\n" + "\n".join(sq) + '" </s>'
            for cq, sq in zip(examples["complex_question"], simple_question_lst)]
    examples["text"] = text
    return examples

dataset = load_from_disk("question_dataset/question_dataset")
train_dataset = dataset["train"]
train_dataset = train_dataset.map(transform, batched=True)
val_dataset = dataset['validation']
val_dataset = val_dataset.map(transform, batched=True)

# Load the base model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0}
)

base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# Load MistralAI tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps, # the number of training steps the model will take
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    do_eval=do_eval,
    evaluation_strategy=evaluation_strategy,
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
    eval_dataset=val_dataset,
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)