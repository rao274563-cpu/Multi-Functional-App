from transformers import pipeline
class NextWordPredictor:
    def __init__(self):
        self.generator = pipeline(
            "text-generation",
            model="gpt2"
        )

    def predict(self, text):
        result = self.generator(
            text,
            max_length=len(text.split()) + 5,
            num_return_sequences = 1,
            do_sample=False
        )   
        generated_text = result[0]['generated_text']

        # Extract only next word
        next_word = generated_text.replace(text, "").strip().split(" ")[0] 
        return next_word
