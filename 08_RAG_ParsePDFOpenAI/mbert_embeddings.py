# https://huggingface.co/google-bert/bert-base-multilingual-cased
# Small model, very fast and runs on my machine
# Multilingual model (104 languages)


from transformers import BertTokenizer, BertModel

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained("bert-base-multilingual-cased")

text = "Replace me by any text you'd like."
# Tokenize text
encoded_input = tokenizer(text, return_tensors='pt')
# Create embeddings
output = model(**encoded_input)
# Print the output
print(output)