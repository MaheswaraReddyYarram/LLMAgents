from transformers import pipeline

if __name__ == '__main__':

    # create classifier
    classifier = pipeline('sentiment-analysis', model = 'cardiffnlp/twitter-roberta-base-sentiment')

    # Array of example texts for classification
    comments = [
        "This product is absolutely amazing. I highly recommend it to everyone!",  # Positive feedback
        "It's okay, but nothing special. I expected more for the price.",  # Neutral feedback
        "Terrible experience. The product broke within a week!",  # Bad feedback
        "Good value for the money. I'm satisfied with this purchase."  # Positive feedback
    ]

    label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

    for text in comments:
        #get result
        result = classifier(text)

        sentiment = label_map[result[0]['label']]

        print(f'Text is :{text}')
        print(f'Sentiment : {sentiment} with score: {round(result[0]["score"], 4)}')
        print("-----------------------------------")
