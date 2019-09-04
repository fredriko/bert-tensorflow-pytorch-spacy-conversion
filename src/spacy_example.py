import spacy
import torch
is_using_gpu = spacy.prefer_gpu()
if is_using_gpu:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

nlp = spacy.load("ru_pytt_rubert_cased")
doc = nlp("Рад познакомиться с вами.")
print(doc.vector)
print(doc[0].similarity(doc[0]))
print(doc[0].similarity(doc[1]))
