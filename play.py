from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("abakirci/admbkrc-turkish-tokenizer")

print(tokenizer)