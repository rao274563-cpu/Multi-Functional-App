# Multifunctional NLP & Image Generation Tool using Hugging Face Models

## 🚀 Project Overview

The **Multifunctional NLP & Image Generation Tool** is an AI-powered application that integrates multiple Natural Language Processing (NLP) and Computer Vision capabilities into a single platform using pretrained Hugging Face models.

The application provides an intuitive interface where users can perform a variety of AI-driven tasks, including text summarization, sentiment analysis, chatbot interaction, question answering, next-word prediction, story generation, and image generation.

This project demonstrates the practical integration of multiple transformer-based machine learning models into a unified application and showcases the capabilities of modern AI systems for real-world use cases.

---

## 📸 Application Preview

### NLP Features
<img width="1000" height="667" alt="Screenshot 2026-06-14 161836" src="https://github.com/user-attachments/assets/61dac036-e20e-400c-aad4-d514ca15262a" />
<img width="1021" height="377" alt="Screenshot 2026-06-14 162054" src="https://github.com/user-attachments/assets/b8d34a06-8f0e-4dab-bec5-4e437ba447a2" />


### Image Generation

<img width="635" height="395" alt="Screenshot 2026-06-14 162141" src="https://github.com/user-attachments/assets/d3befd8b-5284-4f11-8a3b-ef7eddffaf85" />


---

## 🎯 Project Features

### 📝 Text Summarization

Generate concise and meaningful summaries from lengthy text passages using transformer-based summarization models.

### 😊 Sentiment Analysis

Analyze text sentiment and classify it as:

* Positive
* Negative

### 🤖 AI Chatbot

Interact with an intelligent chatbot capable of responding to user queries using pretrained language models.

### ❓ Question Answering

Extract precise answers from a given context using transformer-based question-answering models.

### 🔮 Next Word Prediction

Predict the most probable next word based on user-provided text input.

### 📖 Story Generation

Generate creative and context-aware stories from custom prompts.

### 🎨 Image Generation

Generate images from textual descriptions using Stable Diffusion image-generation models.

---

## 📊 Key Insights

* Successfully integrated multiple Hugging Face models into a single application.
* Demonstrated both NLP and Computer Vision capabilities within one platform.
* Implemented task-specific model pipelines for improved usability and performance.
* Applied prompt engineering and generation controls to enhance output quality.
* Developed a scalable architecture that allows easy integration of future AI features.
* Showcased practical applications of transformer-based models in real-world scenarios.

---

## 🛠 Tools & Technologies

### Programming Language

* Python 3.11

### Machine Learning & AI

* Hugging Face Transformers
* Hugging Face Diffusers
* Stable Diffusion
* PyTorch

### NLP Models

* DistilBART (Text Summarization)
* DistilBERT (Question Answering)
* GPT-2 (Chatbot, Story Generation, Next Word Prediction)
* Hugging Face Sentiment Analysis Pipeline

### Frontend

* Streamlit

### Additional Libraries

* Accelerate
* Safetensors
* Pillow
* NumPy

### Development Environment

* Visual Studio Code
* Windows 11
* Git & GitHub

---

## 📂 Dataset Information

This project primarily utilizes pretrained Hugging Face models and does not require a custom dataset.

## 📁 Project Structure

```text
MultiFunctional_AI_Tool/
│
├── app/
│   ├── main.py
│   └── streamlit_app.py
│
├── models/
│   ├── summarizer.py
│   ├── sentiment.py
│   ├── chatbot.py
│   ├── qa.py
│   ├── next_word.py
│   ├── story.py
│   └── image_gen.py
│
├── screenshots/
│   ├── home.png
│   ├── nlp_features.png
│   └── image_generation.png
│
├── generated_image.png
├── requirements.txt
├── test_setup.py
├── README.md
├── .gitignore
└── venv/
```

---

## 📈 Model Evaluation

### NLP Tasks

Performance can be evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

### User Experience

* Response Relevance
* Generation Quality
* User Satisfaction

### Image Generation

* Visual Quality
* Prompt Adherence
* Generation Time

---

## 💡 Business Use Cases

* AI-Powered Content Creation
* Intelligent Customer Support Systems
* Automated Text Analysis
* Educational AI Assistants
* Creative Writing Assistance
* AI Image Generation Platforms
* Research and Productivity Tools
* Marketing Content Generation

---

## ▶️ Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/rao274563-cpu/MultiFunctional_AI_Tool.git
cd MultiFunctional_AI_Tool
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app/streamlit_app.py
```

---

## 🔮 Future Improvements

* Integration of larger conversational models (Llama, Mistral, Gemma, etc.)
* Real-time Speech-to-Text and Text-to-Speech support
* Multi-language support
* Conversation memory for chatbot interactions
* User authentication and profile management
* Cloud deployment using Docker and AWS/GCP/Azure
* Advanced image generation controls
* Image editing and image-to-image generation
* API integration for third-party applications
* GPU-based performance optimization

---

## 👨‍💻 Author

### Sachin Kumar Rao

GitHub:
https://github.com/rao274563-cpu

LinkedIn:
https://www.linkedin.com/in/sachin-rao-535b0b331/

---

## ⭐ Acknowledgements

Special thanks to:

* Hugging Face
* Transformers Library
* Diffusers Library
* PyTorch
* Streamlit
* Open Source AI Community

for providing the tools and resources that made this project possible.

---

## 📜 License

This project is intended for educational, learning, and portfolio purposes.

Feel free to fork, modify, and enhance the project while providing appropriate attribution.

---

If you found this project useful, consider giving the repository a ⭐ on GitHub.

Streamlit
