from transformers import pipeline
class QuestionAnswering:
    def __init__(self):
        self.qa_pipeline = pipeline(
            "question-answering",
            model = "distilbert-base-cased-distilled-squad"
        )

    def answer(self, question, context):
        result = self.qa_pipeline(
            question=question,
            context=context
        )    
        return result["answer"]