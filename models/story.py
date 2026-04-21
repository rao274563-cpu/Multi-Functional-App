from transformers import pipeline
class StoryGenerator:
    def __init__(self):
        self.generator = pipeline(
            "text-generation",
            model="gpt2"
        )

    def generate(self, prompt):
        structured_prompt = f"Write a simple, clear and meaningful short story about {prompt}. The story should be coherent and not repetitive.\nStory:"

        result = self.generator(
            structured_prompt,
            max_length = 200,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.5
        )  

        story =  result[0]['generated_text']  

        # Clean output 
        story = story.replace(structured_prompt, "").strip()
        return story