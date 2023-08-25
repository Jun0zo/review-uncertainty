# bert embedding test (Korean) with torch
# 2019-05-23

from transformers import BertTokenizer

# Example Korean text data
korean_text = "택배가 엉망이네용 저희집 밑에층에 말도없이 놔두고가고"

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Tokenize the text
tokens = tokenizer.tokenize(korean_text)

print(tokens)
