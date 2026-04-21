from transformers import pipeline

class Chatbot:
    def __init__(self):
        self.generator = pipeline(
            "text-generation",
            model = "gpt2"
        ) 

    def chat(self, prompt):

        # Predefined simple responses (for common inputs)
        simple_responses = {
            "hi": "Hello!",
            "hello": "Hi there!",
            "how are you": "I'm good, what about you?",
            "what is your name": "I am your AI assistant."
        }
        # Normalize input
        clean_prompt = prompt.lower().strip()

        for key in simple_responses:
            if key in clean_prompt:
                return simple_responses[key]

        # Otherwise, use GPT-2
        result = self.generator(
            "Userinput:"+ prompt + "\nBot:",
            max_length=40,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )   

        response = result[0]['generated_text']
        # Remove original prompt from response
        return response.split("Bot:")[-1].strip()