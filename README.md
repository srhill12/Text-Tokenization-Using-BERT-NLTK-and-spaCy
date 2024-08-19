
# Text Tokenization Using BERT, NLTK, and spaCy

This project demonstrates text tokenization using various libraries such as BERT, NLTK, and spaCy. The main focus is on tokenizing text data to explore subword, sentence, and word tokenization.

## Table of Contents
- [Dependencies](#dependencies)
- [BERT Tokenization](#bert-tokenization)
- [NLTK Tokenization](#nltk-tokenization)
- [spaCy Tokenization](#spacy-tokenization)
- [Results](#results)
- [Conclusion](#conclusion)

## Dependencies

To run the code in this project, you'll need to install the following Python libraries:

- [transformers](https://huggingface.co/transformers/)
- [nltk](https://www.nltk.org/)
- [spaCy](https://spacy.io/)

You can install these dependencies using pip:

```bash
pip install transformers nltk spacy
```

You'll also need to download the required NLTK and spaCy datasets:

```python
import nltk
nltk.download('punkt')
nltk.download('reuters')

import spacy
spacy.cli.download("en_core_web_sm")
```

## BERT Tokenization

We start by tokenizing a sample sentence using the BERT tokenizer from the `transformers` package.

### Code:
```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "I am learning about subword tokenization."
subwords = tokenizer.tokenize(text)
print(subwords)
```

### Output:
```python
['i', 'am', 'learning', 'about', 'sub', '##word', 'token', '##ization', '.']
```

## NLTK Tokenization

Next, we demonstrate tokenization using the NLTK library. We tokenize both sentences and words from an article in the Reuters dataset.

### Code:
```python
from nltk.corpus import reuters
from nltk.tokenize import sent_tokenize, word_tokenize

# Download the Reuters and Punkt datasets
import nltk
nltk.download("reuters")
nltk.download('punkt')

# Get an article on 'cocoa'
article = reuters.raw(reuters.fileids(categories='cocoa')[0])

# Sentence tokenization
sentences = sent_tokenize(article)
print(sentences[0])

# Word tokenization
words = word_tokenize(sentences[0])
print(words)
```

### Output:
- **Sentence Tokenization:**
  ```python
  "COCOA EXPORTERS EXPECTED TO LIMIT SALES\n  Major cocoa exporters are likely to limit sales in the weeks ahead..."
  ```

- **Word Tokenization:**
  ```python
  ['COCOA', 'EXPORTERS', 'EXPECTED', 'TO', 'LIMIT', 'SALES', 'Major', 'cocoa', 'exporters', 'are', 'likely', 'to', 'limit', 'sales', 'in', 'the', 'weeks', 'ahead', 'in', 'an', 'effort', 'to', 'boost', 'world', 'prices', ',', 'sources', 'close', 'to', 'a', 'meeting', 'of', 'the', 'Cocoa', 'Producers', 'Alliance', '(', 'CPA', ')', 'said', '.']
  ```

## spaCy Tokenization

Finally, we demonstrate tokenization using spaCy, which provides a more advanced NLP pipeline.

### Code:
```python
import spacy

nlp = spacy.load("en_core_web_sm")
spacy_sent = nlp(sentences[0])
tokens = [token.text for token in spacy_sent]
print(tokens)
```

### Output:
```python
['COCOA', 'EXPORTERS', 'EXPECTED', 'TO', 'LIMIT', 'SALES', '\n  ', 'Major', 'cocoa', 'exporters', 'are', 'likely', 'to', '\n  ', 'limit', 'sales', 'in', 'the', 'weeks', 'ahead', 'in', 'an', 'effort', 'to', 'boost', 'world', '\n  ', 'prices', ',', 'sources', 'close', 'to', 'a', 'meeting', 'of', 'the', 'Cocoa', 'Producers', '\n  ', 'Alliance', '(', 'CPA', ')', 'said', '.']
```

## Results

- **BERT Tokenization:** Subword tokenization splits the words into smaller, meaningful parts, capturing more nuances.
- **NLTK Tokenization:** Basic sentence and word tokenization, suitable for straightforward text processing tasks.
- **spaCy Tokenization:** Advanced tokenization that includes special characters and whitespaces, giving a more granular control over the text.

## Conclusion

This project showcases different approaches to text tokenization, each with its own strengths. Depending on your NLP task, you can choose the tokenizer that best fits your needs.

