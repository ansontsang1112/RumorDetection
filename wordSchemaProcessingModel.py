import nltk


def wordProcessingModel(sentence: str):
    # Delete the punctuations in the sentence
    tokenizer = nltk.RegexpTokenizer(r"\w+")  # Regex Format
    tokenizedSentence = tokenizer.tokenize(sentence.lower())
    meaningfulTokenSet = []

    nlStopWords = nltk.corpus.stopwords.words("english")
    for sentFormatting in tokenizedSentence:
        if sentFormatting not in nlStopWords:
            meaningfulTokenSet.append(sentFormatting)

    return meaningfulTokenSet
