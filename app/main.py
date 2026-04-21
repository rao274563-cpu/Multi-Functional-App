from models.summarizer import Summarizer
from models.sentiment import SentimentAnalyzer
from models.chatbot import Chatbot
from models.qa import QuestionAnswering
from models.next_word import NextWordPredictor
from models.story import StoryGenerator
from models.image_gen import ImageGenerator

def main():
    print("Select task:")
    print("1. Summarization")
    print("2. Sentiment Analysis")
    print("3. Chatbot")
    print("4. Question Answering")
    print("5. Next Word Prediction")
    print("6. Story Generation")
    print("7. Image Generation")

    choice = input("Enter your choice: ")


# if __name__ == "__main__":
#     text = """Artificial Intelligence is rapidaly evolving and transforming industries by
#     enabling machines to learn from data and make inteligent decisions."""

    if choice == '4':
        context = input("\nEnter the context:\n")
        question = input("\nEnter the question:\n")
        qa = QuestionAnswering()
        answer = qa.answer(question, context)
        print("\nAnswer:\n", answer)

    elif choice == '7':
            prompt = input("\nEnter Image Description:\n")
            image_gen = ImageGenerator()
            result = image_gen.generate(prompt)
            print("\nGenerated Image:\n", result)    
    
    else:
        text = input("\nEnter the text: ")
    
        if choice == '1':
           summarizer = Summarizer()
           summary = summarizer.summarize(text)
           print("\nSummary:\n", summary)

        elif choice == '2':
             sentiment_analyzer = SentimentAnalyzer()
             sentiment = sentiment_analyzer.analyze(text)
             print("\Sentiment:\n", sentiment)

        elif choice == '3':
            chatbot = Chatbot()
            result = chatbot.chat(text)
            print("\nChatbot Response:\n", result) 

        elif choice == '5':
            predictor = NextWordPredictor()
            next_word = predictor.predict(text)
            print("\nNext Word Prediction:\n", next_word)

        elif choice == "6":
            story_gen = StoryGenerator()
            story = story_gen.generate(text)
            print("\nGenerated Story:\n", story)    

        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()        