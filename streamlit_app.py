import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st

from models.summarizer import Summarizer
from models.sentiment import SentimentAnalyzer
from models.chatbot import Chatbot
from models.qa import QuestionAnswering
from models.next_word import NextWordPredictor
from models.story import StoryGenerator
from models.image_gen import ImageGenerator

st.title("Multifunctional AI Tool")

task = st.selectbox(
    "Select a task:",
    [
        "Summarization",
        "Sentiment Analysis",
        "Chatbot",
        "Question Answering",
        "Next Word Prediction",
        "Story Generation",
        "Image Generation"
    ]
)

# Text Input
if task != "Question Answering" and task != "Image Generation":
    text = st.text_area("Enter your text:")

# Question Answering
if task == "Question Answering":
    context = st.text_area("Enter context:")
    question = st.text_input("Enter question:")

# Image Generation     
if task == "Image Generation":
    prompt = st.text_input("Enter Image Description:")


if st.button("Run"):

    if task == "Summarization":
        model = Summarizer()
        result = model.summarize(text)
        st.subheader("Summary:")
        st.write(result)

    elif task == "Sentiment Analysis":
        model = SentimentAnalyzer()
        result = model.analyze(text)
        st.subheader("Sentiment:")
        st.write(result)

    elif task == "Chatbot":
        model = Chatbot()
        result = model.chat(text)
        st.subheader("Chatbot Response:")
        st.write(result)     

    elif task == "Question Answering":
        model = QuestionAnswering()
        result = model.answer(question, context)
        st.subheader("Answer:")
        st.write(result)

    elif task == "Next Word Prediction":
        model = NextWordPredictor()    
        result = model.predict(text)
        st.subheader("Next Word Prediction")
        st.write(result)

    elif task == "Story Generation":
        model = StoryGenerator()
        result = model.generate(text)
        st.subheader("Generated Story:")
        st.write(result)

    elif task == "Image Generation":
        model = ImageGenerator()   
        result = model.generate(prompt)
        st.subheader("Image generated:")
        st.image("generated_image.png") 
