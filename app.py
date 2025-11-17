import streamlit as st
from backened import *

st.title("🧠 Multifunctional NLP & Image Generation Tool")

task = st.selectbox(
    "Select a task:",
    [
        "Select a task",
        "Text Summarization",
        "Next Word Prediction",
        "Story Generation",
        "Chatbot",
        "Sentiment Analysis",
        "Question Answering",
        "Image Generation"
    ]
)


if task == "Text Summarization":
    text = st.text_area("Enter text for summarization:")
    if st.button("Summarize"):
        st.write(summarize_text(text))

elif task == "Next Word Prediction":
    text = st.text_input("Enter some text:")
    if st.button("Predict"):
        st.write(next_word_predict(text))

elif task == "Story Generation":
    prompt = st.text_area("Enter story prompt:")
    if st.button("Generate"):
        st.write(generate_story(prompt))

elif task == "Chatbot":
    user_input = st.text_input("Say something to the chatbot:")
    if st.button("Ask"):
        st.write(chat_with_bot(user_input))

elif task == "Sentiment Analysis":
    text = st.text_input("Enter sentence:")
    if st.button("Analyze"):
        st.write(analyze_sentiment(text))

elif task == "Question Answering":
    context = st.text_area("Enter context paragraph:")
    question = st.text_input("Enter your question:")
    if st.button("Answer"):
        st.write(answer_question(question, context))

elif task == "Image Generation":
    prompt = st.text_input("Describe the image to generate:")
    if st.button("Generate Image"):
        path = generate_image(prompt)
        st.image(path)
