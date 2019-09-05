## How to convert a BERT model from Tensorflow to PyTorch and spaCy

This repository contains instructions and sample code for converting a BERT Tensorflow model
to work with Hugging Face's [pytorch-transformers](https://github.com/huggingface/pytorch-transformers)
and as a package for explosion.ai's [spaCy](https://spacy.io/) via [spacy-pytorch-transformers](https://github.com/explosion/spacy-pytorch-transformers). 

The instructions use the Russian BERT model (RuBERT) created by [DeepPavlov](https://deeppavlov.ai) as working example.

### Pre-requisites

You need the following software installed on your computer to be able to install and run the examples in this guide.

* Git
* Python 3.6 or later
* pip3
* virtualenv

### Download the example BERT model

Download [RuBERT](http://docs.deeppavlov.ai/en/master/features/pretrained_vectors.html#bert) from http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_v1.tar.gz

For the sake of this example, place the downloaded RuBERT file in your user's root directory, and unpack it with

```
tar zxvf rubert_cased_L-12_H-768_A-12_v1.tar.gz
```

The unpacked model is now available in `~/rubert_cased_L-12_H-768_a-12_v1/`


### Set-up the working environment

Clone this repository, create a virtual environment, and install the dependencies by giving the following commands in a shell:

```
git clone https://github.com/fredriko/bert-tensorflow-pytorch-spacy-conversion.git
cd bert-tensorflow-pytorch-spacy-conversion
virtualenv -p python3 ~/venv/bert-tensorflow-pytorch-spacy-conversion
source ~/venv/bert-tensorflow-pytorch-spacy-conversion/bin/activate
pip3 install -r requirements.txt
```


### Convert the BERT Tensorflow model to work with Hugging Face's pytorch-transformers

Convert the Tensorflow RuBERT model to a PyTorch equivalent with this command:

```
$ python3 -m pytorch_transformers.convert_tf_checkpoint_to_pytorch \
--tf_checkpoint_path ~/rubert_cased_L-12_H-768_A-12_v1/bert_model.ckpt.index \ 
--bert_config_file ~/rubert_cased_L-12_H-768_A-12_v1/bert_config.json \
--pytorch_dump_path ~/rubert_cased_L-12_H-768_A-12_v1/pytorch_model.bin
```

After the conversion, copy the required files to a separate directory; `~/pytorch-rubert/`:
```
mkdir ~/pytorch-rubert
cp ~/rubert_cased_L-12_H-768_A-12_v1/rubert_pytorch.bin ~/pytorch-rubert/.
cp ~/rubert_cased_L-12_H-768_A-12_v1/vocab.txt ~/pytorch-rubert/.
cp ~/rubert_cased_L-12_H-768_A-12_v1/bert_config.json ~/pytorch-rubert/config.json
```

You now have the files required to use RuBERT in pytorch-transformers. The following code snippet is an example of how the PyTorch model can be loaded and used in pytorch-transformers ([source])(src/pytorch_transformers_example.py):

```
import torch
from pytorch_transformers import *
from pathlib import Path

sample_text = "Рад познакомиться с вами."
my_model_dir = str(Path.home() / "pytorch-rubert")

tokenizer = BertTokenizer.from_pretrained(my_model_dir)
model = BertModel.from_pretrained(my_model_dir, output_hidden_states=True)

input_ids = torch.tensor([tokenizer.encode(sample_text, add_special_tokens=True)])
print(f"Input ids: {input_ids}")
with torch.no_grad():
    last_hidden_states = model(input_ids)[0]
    print(f"Shape of last hidden states: {last_hidden_states.shape}")
    print(last_hidden_states)
```

### Convert the pytorch-transformer model to a spaCy package

In order to create a spaCy package of the PyTorch model, it first has to be saved to disk
as a serialized pipeline. First, create the directory in which to save the pipeline, then run
the [script](src/serialize_spacy_nlp_pipeline.py) for serializing and saving it.

```
mkdir ~/spacy-rubert
python3 -m src.serialize_spacy_nlp_pipeline
```

You now have all you need to create a spaCy package in `~/spacy-rubert`. 

**OPTIONAL:** fill in the appropriate information in `~/spacy-rubert/meta.json` 
before proceeding.

Run the following commands to create a spaCy package from the serialized pipeline and save it to `~/spacy-rubert-package`:

```
mkdir ~/spacy-rubert-package
python3 -m spacy package ~/spacy-rubert ~/spacy-rubert-package
```
**NOTE:** that the name of the model directory under `~/spacy-rubert-package` depends on the 
information you supplied in `~/spacy-rubert/meta.json` in the previous step. The name used below
originates from a raw `meta.json` file.
```
cd ~/spacy-rubert-package/ru_model-0.0.0
python3 setup.py sdist
```

After successful completion of the above commands, the RuBERT model is available as a spaCy package in:

```
~/spacy-rubert-package/ru_model-0.0.0/dist/ru_model-0.0.0.tar.gz
```

Install it with:

```
pip3 install ~/spacy-rubert-package/ru_model-0.0.0/dist/ru_model-0.0.0.tar.gz
```

Verify its presence in the current virtualenv:

```
pip3 freeze | grep ru-model
> ru-model==0.0.0
```

Here is an example of how the package can be loaded and used ([source](src/spacy_example.py)):

```
import spacy

nlp = spacy.load("ru_model")
doc = nlp("Рад познакомиться с вами.")
print(doc.vector)
print(doc[0].similarity(doc[0]))
print(doc[0].similarity(doc[1]))
```

**NOTE:** that the above example does not make use of a GPU. For that to happen, 
you need a different installation of spaCy than the one specified in the `requirements.txt`
in this repository.
 
 