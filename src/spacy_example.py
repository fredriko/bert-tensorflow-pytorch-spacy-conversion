import spacy

nlp = spacy.load("ru_model")
doc = nlp("Рад познакомиться с вами.")
print(doc.vector)
print(doc[0].similarity(doc[0]))
print(doc[0].similarity(doc[1]))
