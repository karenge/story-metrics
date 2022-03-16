import numpy as np
import tqdm
import pandas as pd
import torch
import torch.nn as nn

torch.cuda.empty_cache()

import gc

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
#import matplotlib

# specify GPU
device = torch.device("cuda")

df = pd.read_csv("final_data.csv", on_bad_lines='skip', quotechar='"', engine='python')

df['text'] = df['sentence1']+' '+df['sentence2']+' '+df['sentence3']+' '+df['sentence4']+' '+df['sentence5']
df['comet'] = pd.DataFrame(np.c_[np.vstack(df['causes1']),np.vstack(df['causes2']),np.vstack(df['causes3']),np.vstack(df['causes4']),np.vstack(df['causes5']),np.vstack(df['hasprereq1']),np.vstack(df['hasprereq2']),np.vstack(df['hasprereq3']),np.vstack(df['hasprereq4']),np.vstack(df['hasprereq5'])])
df['emotions'] = pd.DataFrame(np.c_[np.vstack(df['emotions1']),np.vstack(df['emotions2']),np.vstack(df['emotions3']),np.vstack(df['emotions4']),np.vstack(df['emotions5'])])

#print(torch.cuda.memory_summary())

X = df.loc[:, df.columns == 'text','comet','emotions']
y = df.loc[:, df.columns == 'label']

train, test, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=1)

train, val, train_labels, val_labels = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

print(train.head())
#[id, text, label]
# split train dataset into train, validation and test sets
train_text = train['text']
train_comet = train['comet']
train_emo = train['emotions']
train_labels = train['label']

test_text = test['text']
test_comet = test['comet']
test_emo = test['emotions']
test_labels = test['label']

val_text = val['text']
val_comet = val['comet']
val_emo = val['emotions']
val_labels = val['label']

'''
# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')
#bert = AutoModel.from_pretrained('checkpoints/...')

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# get length of all the messages in the train set
seq_len = [len(i.split()) for i in train_text]

#pd.Series(seq_len).hist(bins = 30)

# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = 100,
    pad_to_max_length=True,
    truncation=True
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = 100,
    pad_to_max_length=True,
    truncation=True
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = 100,
    pad_to_max_length=True,
    truncation=True
)
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())
train_comet = torch.tensor(train_comet.tolist())
train_emo = torch.tensor(train_emo.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())
val_comet = torch.tensor(val_comet.tolist())
val_emo = torch.tensor(val_emo.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())
test_comet = torch.tensor(test_comet.tolist())
test_emo = torch.tensor(test_emo.tolist())

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#define a batch size
batch_size = 32

# wrap tensors
#train_data = TensorDataset(train_seq, train_mask, train_y)
train_data = TensorDataset(train_seq, train_mask, train_comet, train_emo ,train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_comet, val_emo, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

class BERT_Arch(nn.Module):

    def __init__(self, bert):

      super(BERT_Arch, self).__init__()

      self.bert = bert

      # dropout layer
      self.dropout = nn.Dropout(0.1)
      # relu activation function
      self.relu =  nn.ReLU()


      # dense layer 1
      self.fc1 = nn.Linear(768,512)
      # dense layer 2 (Output layer)

      self.num_emo = 8
      self.num_sents = 5
      self.num_rels = 2
      self.embed_size = 300

      self.emo = nn.Linear(self.num_emo * self.num_sents, 256)
      self.comet = nn.Linear(self.embed_size * self.num_sents * self.num_rels, 256)

      self.fc2 = nn.Linear(1024,2)
      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, comets, emos, mask):

      #pass the inputs to the model
      _, cls_hs = self.bert(sent_id, attention_mask=mask,return_dict=False)

      # how does br


      c = self.comet(comets)
      e = self.emo(emos)

      x = self.fc1(cls_hs)

      x = torch.cat((x, c, e), dim = 1)

      x = self.relu(x)

      x = self.dropout(x)

      # output layer
      x = self.fc2(x)
      # apply softmax activation
      x = self.softmax(x)

      return x

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)
gc.collect()

# optimizer from hugging face transformers
from transformers import AdamW

# define the optimizer
optimizer = AdamW(model.parameters(),
                  lr = 1e-5)          # learning rate

from sklearn.utils.class_weight import compute_class_weight

#compute the class weights
class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)

print("Class Weights:",class_weights)

# converting list of class weights to a tensor
weights= torch.tensor(class_weights,dtype=torch.float)

# push to GPU
weights = weights.to(device)

# define the loss function
cross_entropy  = nn.NLLLoss()

# number of training epochs
epochs = 10

# function to train the model
def train():

  model.train()

  total_loss, total_accuracy = 0, 0

  # empty list to save model predictions
  total_preds=[]

  # iterate over batches
  for step,batch in enumerate(train_dataloader):

    # progress update after every 50 batches.
    if step % 50 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

    # push the batch to gpu
    batch = [r.to(device) for r in batch]

    sent_id, mask, comets, emos, labels = batch

    # clear previously calculated gradients
    model.zero_grad()

    # get model predictions for the current batch
    preds = model(sent_id, mask)

    # compute the loss between actual and predicted values
    loss = cross_entropy(preds, labels)

    # add on to the total loss
    total_loss = total_loss + loss.item()

    # backward pass to calculate the gradients
    loss.backward()

    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # update parameters
    optimizer.step()

    # model predictions are stored on GPU. So, push it to CPU
    preds=preds.detach().cpu().numpy()

    # append the model predictions
    total_preds.append(preds)

  # compute the training loss of the epoch
  avg_loss = total_loss / len(train_dataloader)

  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  #returns the loss and predictions
  return avg_loss, total_preds
'''

'''
# function for evaluating the model
def evaluate():

  print("\nEvaluating...")

  # deactivate dropout layers
  model.eval()

  total_loss, total_accuracy = 0, 0

  # empty list to save the model predictions
  total_preds = []

  # iterate over batches
  for step,batch in enumerate(val_dataloader):

    # Progress update every 50 batches.
    if step % 50 == 0 and not step == 0:

      # Report progress.
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

    # push the batch to gpu
    batch = [t.to(device) for t in batch]

    sent_id, mask, comets, emos, labels = batch

    # deactivate autograd
    with torch.no_grad():

      # model predictions
      preds = model(sent_id, comets, emos, mask)

      # compute the validation loss between actual and predicted values
      loss = cross_entropy(preds,labels)

      total_loss = total_loss + loss.item()

      preds = preds.detach().cpu().numpy()

      total_preds.append(preds)

  # compute the validation loss of the epoch
  avg_loss = total_loss / len(val_dataloader)

  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  return avg_loss, total_preds

# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

#for each epoch
for epoch in range(epochs):

    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

    #train model
    train_loss, _ = train()

    #evaluate model
    valid_loss, _ = evaluate()

    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')

    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

print(torch.cuda.memory_summary())

def predict():
    #load weights of best model
    path = 'saved_weights.pt'
    model.load_state_dict(torch.load(path))

    print("model loaded")

    LEN = 32
    iter = 0
    predictions = []
    # get predictions for test data
    with torch.no_grad():
        for iter in range(len(test_seq)//LEN):
            preds = model(test_seq[iter*LEN:(iter+1)*LEN].to(device), test_mask[iter*LEN:(iter+1)*LEN].to(device))
            preds = preds.detach().cpu().numpy()

            torch.cuda.clear_cache()

            predictions.append(np.argmax(preds, axis = 1))
            print(classification_report(test_y[iter*LEN:(iter+1)*LEN], preds))
predict()
'''
