from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.classifier = pipeline(
            "sentiment-analysis",
        )

    def analyze(self,text):
        result = self.classifier(text)[0]

        label = result['label']
        score = result['score']

        if score < 0.7:
            return {'label': 'Neutral', 'score': score}
        return {'label': label, 'score': score}   