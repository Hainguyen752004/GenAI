#  Fine-Tuning DeepSeek-R1-Distill-Qwen-1.5B on Vietnamese Philosophy QA
##  Model & Tokenizer
base model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B.
Tokenizer: pad_token is set to eos_token to prevent padding-related errors
##  Dataset
Source: Vietnamese-Philosophy-QA.
If no test set is present, the training set is split 90/10 for training and evaluation.
Only keeps entries with "question_type" == "sentences".
##  Training Setup
Training is handled via Hugging Face Trainer with:
per_device_train_batch_size = 4
gradient_accumulation_steps = 8
learning_rate = 2e-5
num_train_epochs = 200
Logging and evaluation at each epoch
Mixed precision (FP16/BFloat16) enabled if supported
# Main.py 
Run the script: python main.py 
