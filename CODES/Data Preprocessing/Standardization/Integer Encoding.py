import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

docs = ['the cat sat quietly on the windowsill',
    'she bought fresh apples from the market',
    'rain started falling just after noon',
    'he forgot his umbrella at the office',
    'the movie starts at eight tonight',
    'my laptop battery died during the meeting',
    'they hiked up the mountain before sunrise',
    'coffee tastes better with a little sugar',
    'the train was delayed by twenty minutes',
    'she is learning to play the guitar',
    'the bakery smells amazing in the morning',
    'we watched the sunset from the rooftop',
    'his car needed an oil change last week',
    'the library closes early on Sundays',
    'birds were chirping loudly in the garden',
    'she finished the report before the deadline',
    'the kids built a sandcastle at the beach',
    'he ordered pizza for the whole team',
    'the road was blocked due to construction',
    'they planted new flowers in the backyard']

tokenizer = Tokenizer(oov_token = "<nothing>")

tokenizer.fit_on_texts(docs)
# print(tokenizer.word_index)
# print(tokenizer.document_count)

sequences = tokenizer.texts_to_sequences(docs)
sequences = pad_sequences(sequences, padding="post")

print(sequences)