from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    AutoModelForCausalLM, pipeline,
)
from diffusers import StableDiffusionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# BACKEND FUNCTIONS

# 1. Summarization Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
def summarize_text(text):
    return summarizer(text)[0]['summary_text']

# 2. Next Word Prediction (GPT-2)
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
def next_word_predict(text):
    inputs = gpt2_tokenizer(text, return_tensors="pt").to(device)
    outputs = gpt2_model.generate(
        inputs["input_ids"],
        max_length=50,
        do_sample=True,
        top_k=50,
    )
    return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

# 3. Story Generator
story_model = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device)
story_tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
def generate_story(prompt):
    inputs = story_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = story_model.generate(
        inputs["input_ids"],
        max_length=150,
        temperature=0.9,
        do_sample=True
    )
    return story_tokenizer.decode(outputs[0], skip_special_tokens=True)

# 4. Chatbot - DialoGPT
chatbot_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
chatbot_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").to(device)
def chat_with_bot(text):
    inputs = chatbot_tokenizer.encode(text + chatbot_tokenizer.eos_token, return_tensors="pt").to(device)
    outputs = chatbot_model.generate(inputs, max_length=1000, pad_token_id=chatbot_tokenizer.eos_token_id)
    return chatbot_tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)

# 5. Sentiment Analysis
sentiment = pipeline("sentiment-analysis")
def analyze_sentiment(text):
    return sentiment(text)[0]

# 6. Question Answering
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
def answer_question(question, context):
    return qa_pipeline({"question": question, "context": context})["answer"]

# 7. Image Generation
image_pipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)
image_pipeline = image_pipeline.to(device)
def generate_image(prompt):
    image = image_pipeline(prompt).images[0]
    image.save("generated.png")
    return "generated.png"
