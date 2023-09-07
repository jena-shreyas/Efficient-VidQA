from modeling_t5 import T5EncoderModel
from transformers import T5TokenizerFast

tokenizer = T5TokenizerFast.from_pretrained("t5-large")
model = T5EncoderModel.from_pretrained("t5-large")
input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
outputs = model(input_ids=input_ids)
last_hidden_states = outputs.last_hidden_state