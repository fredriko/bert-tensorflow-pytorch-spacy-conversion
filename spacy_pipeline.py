from spacy_pytorch_transformers import PyTT_Language, PyTT_WordPiecer, PyTT_TokenVectorEncoder

name = "ru_pytt_rubert_cased"
path = "/Users/fredriko/Dropbox/data/models/rubert/pytorch-rubert/"

nlp = PyTT_Language(pytt_name=name, meta={"lang": "ru"})
nlp.add_pipe(nlp.create_pipe("sentencizer"))
nlp.add_pipe(PyTT_WordPiecer.from_pretrained(nlp.vocab, path))
nlp.add_pipe(PyTT_TokenVectorEncoder.from_pretrained(nlp.vocab, path))
print(nlp.pipe_names)
nlp.to_disk("/tmp/rubert_spacy/")
