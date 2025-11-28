from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer (first time downloads ~1GB)
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_local(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
result = generate_local("What is the capital of France?")
print("Result:", result)