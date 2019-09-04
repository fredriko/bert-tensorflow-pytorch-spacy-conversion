from spacy_pytorch_transformers import PyTT_Language, PyTT_WordPiecer, PyTT_TokenVectorEncoder
from pathlib import Path

pytorch_path = str(Path.home() / "pytorch-rubert")
spacy_path = str(Path.home() / "spacy-rubert")
name = "ru_pytt_rubert_cased"

nlp = PyTT_Language(pytt_name=name, meta={"lang": "ru"})
nlp.add_pipe(nlp.create_pipe("sentencizer"))
nlp.add_pipe(PyTT_WordPiecer.from_pretrained(nlp.vocab, pytorch_path))
nlp.add_pipe(PyTT_TokenVectorEncoder.from_pretrained(nlp.vocab, pytorch_path))
print(nlp.pipe_names)
nlp.to_disk(spacy_path)
