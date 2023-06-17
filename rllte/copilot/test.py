from transformers import T5ForConditionalGeneration, AutoTokenizer

checkpoint = "./codet5p-770m-py"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

while True:
    prompt = input("Prompt: ")
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=1000)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# ==>     print('Hello World!')