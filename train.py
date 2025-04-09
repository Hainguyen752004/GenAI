import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Khai báo model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Đặt pad_token để tránh lỗi

# Load dataset
raw_dataset = load_dataset("Trongdz/Vietnamese-Philosophy-QA")
if "test" not in raw_dataset:  # Nếu không có tập test, tự chia dữ liệu
    dataset = raw_dataset["train"].train_test_split(test_size=0.1)
else:
    dataset = raw_dataset

# Lọc dữ liệu chỉ lấy question_type == "sentences"
def filter_sentences(example):
    return example["question_type"] == "sentences"

dataset["train"] = dataset["train"].filter(filter_sentences)
dataset["test"] = dataset["test"].filter(filter_sentences)

# Tokenize dữ liệu
def tokenize_function(examples):
    inputs = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples["question"], examples["answer"])]
    return tokenizer(inputs, padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Cấu hình LoRA
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
)

# Load model với LoRA
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch_dtype).to(device)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()  # Bật Flash Attention nếu có

# Data Collator để tối ưu batch
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # Không dùng Masked LM, chỉ cần Causal LM
)

# Cấu hình TrainingArguments
logging_steps = max(1, len(tokenized_datasets["train"]) // 40)  # Giảm log quá nhiều
training_args = TrainingArguments(
    output_dir="./qwen-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=200,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=logging_steps,
    fp16=True if torch_dtype == torch.float16 else False,
    report_to="none"
)

# Khởi tạo Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator
)

# Fine-tune
trainer.train()

# Lưu model đã fine-tune (chỉ lưu adapter LoRA)
model.save_pretrained("./qwen-finetuned", save_adapter=True)
tokenizer.save_pretrained("./qwen-finetuned")