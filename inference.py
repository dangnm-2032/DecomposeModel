from transformers import BloomForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from datasets import Dataset, load_dataset, load_from_disk


### VietCuna
def first_evaluate(complex_question):
    eval_prompt = '''Bạn là trợ lý trí tuệ nhân tạo hỗ trợ người dùng về các vấn đề về luật. Người dùng nhập câu hỏi phức tạp có liên quan đến luật, bạn sẽ phân tách câu hỏi phức tạp thành nhiều câu hỏi đơn giản. Mỗi câu hỏi đơn giản được sinh ra thể hiện từng vấn đề con liên quan đến luật của câu hỏi phức tạp để có thể sử dụng các kiến thức về luật trả lời cho từng câu hỏi đơn giản đó.\n\n### Human: {human_message}\n\n### Assistant:"'''

    input_text = eval_prompt.format(human_message=complex_question)
    model_input = tokenizer(input_text, return_tensors="pt").to("cuda")

    model.eval()
    with torch.no_grad():
        output = tokenizer.decode(model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[0], skip_special_tokens=True)
    return output

def oneshot_evaluate(complex_question):
    eval_prompt = '''Bạn là trợ lý trí tuệ nhân tạo hỗ trợ người dùng về các vấn đề về luật.
Người dùng nhập câu hỏi phức tạp có liên quan đến luật, bạn sẽ phân tách câu hỏi phức tạp thành nhiều câu hỏi đơn giản.
Mỗi câu hỏi đơn giản được sinh ra thể hiện từng vấn đề con liên quan đến luật của câu hỏi phức tạp để có thể sử dụng các kiến thức về luật trả lời cho từng câu hỏi đơn giản đó.
### Human: Cho hỏi Kế toán viên trong cơ quan hành chính nhà nước phải có trình độ như thế nào?
### Assistant:"
1. Kế toán viên trong cơ quan hành chính nhà nước có phạm vi quyền hạn như thế nào?
2. Kế toán viên trong cơ quan hành chính nhà nước phải đáp ứng điều kiện về trình độ như thế nào?
3. Công việc của Kế toán viên trong cơ quan hành chính nhà nước là gì?"

### Human: {human_message}
### Assistant:"'''
    input_text = eval_prompt.format(human_message=complex_question)
    model_input = tokenizer(input_text, return_tensors="pt").to("cuda")

    model.eval()
    with torch.no_grad():
        output = tokenizer.decode(model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[0], skip_special_tokens=True)
    return output

model_name = 'merged_model'

model = BloomForCausalLM.from_pretrained(model_name).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_from_disk("question_dataset")
train_dataset = dataset["train"]

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

train_dataset = train_dataset.map(transform, batched=True)
val_dataset = dataset['validation']
val_dataset = val_dataset.map(transform, batched=True)

print("FIRST EVALUATE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
complex_question = val_dataset[0]["complex_question"]
print(first_evaluate(complex_question))
print(val_dataset[0]["triplets"])
print()

complex_question = val_dataset[1]["complex_question"]
print(first_evaluate(complex_question))
print(val_dataset[1]["triplets"])
print()

complex_question = val_dataset[2]["complex_question"]
print(first_evaluate(complex_question))
print(val_dataset[2]["triplets"])
print()

complex_question = val_dataset[-1]["complex_question"]
print(first_evaluate(complex_question))
print(val_dataset[-1]["triplets"])
print()

print("SECOND EVALUATE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
complex_question = val_dataset[0]["complex_question"]
print(oneshot_evaluate(complex_question))
print(val_dataset[0]["triplets"])
print()

complex_question = val_dataset[1]["complex_question"]
print(oneshot_evaluate(complex_question))
print(val_dataset[1]["triplets"])
print()

complex_question = val_dataset[2]["complex_question"]
print(oneshot_evaluate(complex_question))
print(val_dataset[2]["triplets"])
print()

complex_question = val_dataset[-1]["complex_question"]
print(oneshot_evaluate(complex_question))
print(val_dataset[-1]["triplets"])
print()