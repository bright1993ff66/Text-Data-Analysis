# Why text analysis?

1. sentiment analysis
2. spam filtering
3. plagariasm detection/document similarity
4. phrase extraction, summarization

Here is an easy example:

```python
from nltk.tokenize import sent_tokenize
sent_tokenize("Today is a great day, isn't it?")
```
From the code above, we can split a text into seperate sentences.
