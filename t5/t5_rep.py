import os
import torch
import argparse
# from tqdm import tqdm
# from torch.optim import Adam
# # import evaluate  # Bleu
# from torch.utils.data import Dataset, DataLoader, RandomSampler
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5ForConditionalGeneration, T5TokenizerFast

# import warnings
# warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('-n', '--num_epochs', help='Number of training epochs', default=5)
parser.add_argument('-c', '--ckpt_path', help='Checkpoint path', required=True)

args = parser.parse_args()

# class QA_Dataset(Dataset):
#     def __init__(self, tokenizer, dataframe, q_len, t_len):
#         self.tokenizer = tokenizer
#         self.q_len = q_len
#         self.t_len = t_len
#         self.data = dataframe
#         self.questions = self.data["question"]
#         self.context = self.data["context"]
#         self.answer = self.data['answer']

#     def __len__(self):
#         return len(self.questions)

#     def __getitem__(self, idx):
#         question = self.questions[idx]
#         context = self.context[idx]
#         answer = self.answer[idx]

#         question_tokenized = self.tokenizer(question, context, max_length=self.q_len, padding="max_length",
#                                                     truncation=True, pad_to_max_length=True, add_special_tokens=True)
#         answer_tokenized = self.tokenizer(answer, max_length=self.t_len, padding="max_length",
#                                           truncation=True, pad_to_max_length=True, add_special_tokens=True)

#         labels = torch.tensor(answer_tokenized["input_ids"], dtype=torch.long)
#         labels[labels == 0] = -100

#         return {
#             "input_ids": torch.tensor(question_tokenized["input_ids"], dtype=torch.long),
#             "attention_mask": torch.tensor(question_tokenized["attention_mask"], dtype=torch.long),
#             "labels": labels,
#             "decoder_attention_mask": torch.tensor(answer_tokenized["attention_mask"], dtype=torch.long)
#         }

CKPT_PATH = args.ckpt_path
best_path = CKPT_PATH + "best/"
os.makedirs(best_path, exist_ok=True)

try:
    MODEL = T5ForConditionalGeneration.from_pretrained(best_path + "qa_model/")
    TOKENIZER = T5TokenizerFast.from_pretrained(best_path + "qa_tokenizer/")
except:
    MODEL = T5ForConditionalGeneration.from_pretrained("t5-large", return_dict=True) # large
    TOKENIZER = T5TokenizerFast.from_pretrained("t5-large")  # large

# OPTIMIZER = Adam(MODEL.parameters(), lr=0.00001)
Q_LEN = 256   # Question Length
# T_LEN = 32    # Target Length
# BATCH_SIZE = 2
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# NUM_EPOCHS = int(args.num_epochs)

# data = pd.read_csv('/scratch/jenas/dataset3/data.csv')
# data = data.applymap(str)

data = pd.read_csv('/scratch/jenas/dataset3/replace.csv')
data = data.applymap(str)

# trainingset, test_data = train_test_split(data, test_size=0.1, random_state=42)
# train_data, val_data = train_test_split(trainingset, test_size=0.25, random_state=42)


# train_sampler = RandomSampler(train_data.index)
# val_sampler = RandomSampler(val_data.index)
# test_sampler = RandomSampler(test_data.index)

# qa_dataset = QA_Dataset(TOKENIZER, data, Q_LEN, T_LEN)

# train_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
# val_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

MODEL.to(DEVICE)
# min_val_loss = 1e12

# for epoch in range(NUM_EPOCHS):
#     MODEL.train()
#     train_loss = 0
#     val_loss = 0
#     train_batch_count = 0
#     val_batch_count = 0
#     for batch in tqdm(train_loader, desc="Training batches"):
#         input_ids = batch["input_ids"].to(DEVICE)
#         attention_mask = batch["attention_mask"].to(DEVICE)
#         labels = batch["labels"].to(DEVICE)
#         decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

#         outputs = MODEL(
#                           input_ids=input_ids,
#                           attention_mask=attention_mask,
#                           labels=labels,
#                           decoder_attention_mask=decoder_attention_mask
#                         )

#         OPTIMIZER.zero_grad()
#         outputs.loss.backward()
#         OPTIMIZER.step()
#         train_loss += outputs.loss.item()
#         train_batch_count += 1

#     #Evaluation
#     MODEL.eval()
#     for batch in tqdm(val_loader, desc="Validation batches"):
#         input_ids = batch["input_ids"].to(DEVICE)
#         attention_mask = batch["attention_mask"].to(DEVICE)
#         labels = batch["labels"].to(DEVICE)
#         decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

#         outputs = MODEL(
#                           input_ids=input_ids,
#                           attention_mask=attention_mask,
#                           labels=labels,
#                           decoder_attention_mask=decoder_attention_mask
#                         )

#         OPTIMIZER.zero_grad()
#         outputs.loss.backward()
#         OPTIMIZER.step()
#         val_loss += outputs.loss.item()
#         val_batch_count += 1

#     print(f"{epoch+1}/{NUM_EPOCHS} -> Train loss: {train_loss / train_batch_count}\tValidation loss: {val_loss/val_batch_count}")

#     epoch_path = CKPT_PATH + f"{epoch+1}/"
#     os.makedirs(epoch_path, exist_ok=True)
#     MODEL.save_pretrained(epoch_path + "qa_model")
#     TOKENIZER.save_pretrained(epoch_path + "qa_tokenizer")

#     if ((val_loss/val_batch_count) < min_val_loss):
#         min_val_loss = val_loss/val_batch_count
#         MODEL.save_pretrained(best_path + "qa_model")
#         TOKENIZER.save_pretrained(best_path + "qa_tokenizer")

# MODEL.eval()

def accuracy_metric(data, filename: str) -> float :
  context = list(data['context'])
  question = list(data['question'])
  ref_answer = list(data['answer'])
  c=0

  with open(filename, 'w') as f:
    for i in range(len(context)):
        inputs = TOKENIZER(question[i], context[i], max_length=Q_LEN, padding='max_length', truncation=True, add_special_tokens=True)

        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to(DEVICE).unsqueeze(0)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to(DEVICE).unsqueeze(0)

        outputs = MODEL.generate(input_ids = input_ids, attention_mask = attention_mask)

        predicted_answer = TOKENIZER.decode(outputs.flatten(), skip_special_tokens=True)
        predicted_answer = predicted_answer.lower()

        ref_answer[i] = ref_answer[i].lower()

        if predicted_answer == ref_answer[i]:
            c=c+1

        else:
            analysis = context[i]+","+ question[i]+ ","+ ref_answer[i] + "," + predicted_answer+'\n'
            f.write(analysis)

  accuracy = c/len(context)
  accuracy = float(format(accuracy, ".4f"))
  return accuracy

print("\nComputing test metrics ...\n")

analysis_filename = "mistakes_rep.csv"
accuracy = accuracy_metric(data, analysis_filename)

print(f"Test accuracy : {accuracy}")
