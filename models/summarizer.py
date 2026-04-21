from transformers import pipeline

class Summarizer:
    def __init__ (self):
        #Load model once 
        print("Loading Model...")
        self.summarizer = pipeline(
            "summarization",
            model = "sshleifer/distilbart-cnn-12-6"
        )

    def summarize(self, text):
        result = self.summarizer(
            text,
            max_length=30,
            min_length=10,
            do_sample=False
        )
        return result[0]['summary_text']        