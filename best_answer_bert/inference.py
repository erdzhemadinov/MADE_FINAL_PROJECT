# Weights are here
# https://drive.google.com/drive/folders/1kzMThaKGSgKfcdgWFB9YXq8BD5-YKsu8?usp=sharing

import argparse
from torch import no_grad, max, device, cuda, load, tensor, manual_seed, save, nn
from re import sub, escape
from string import punctuation
from nltk.tokenize import RegexpTokenizer
from pandas import read_csv

import time
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# text preprocessing helper functions

from models.bertolet import Embedder, compute_metrics, print_metrics
from sklearn.model_selection import train_test_split
from torch import cat
import torch
import torch.optim as optim
import os

from pytorch_lightning.metrics.functional.classification import auroc
from datetime import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#torch.set_default_tensor_type('torch.DoubleTensor')

class PreProcess:
    @staticmethod
    def clean_text(text):
        """Make text lowercase, remove text in square brackets,remove links,remove punctuation
        and remove words containing numbers."""

        text = text.lower()
        text = sub('\[.*?\]', '', text)
        text = sub('https?://\S+|www\.\S+', '', text)
        text = sub('<.*?>+', '', text)
        text = sub('[%s]' % escape(punctuation), '', text)
        text = sub('\n', '', text)
        text = sub('\w*\d\w*', '', text)
        return text

    @staticmethod
    def text_preprocessing(text):
        """
        Cleaning and parsing the text.

        """
        tokenizer = RegexpTokenizer(r'\w+')
        nopunc = PreProcess.clean_text(text)
        tokenized_text = tokenizer.tokenize(nopunc)
        combined_text = ' '.join(tokenized_text)
        return combined_text

    @staticmethod
    def get_dev(device):
        """
        Get device information
        :return:
        """
        if device is not None:
            device = device
        else:
            device = 'cuda' if cuda.is_available() else 'cpu'

        return device

    @staticmethod
    def import_data(data, t_size, max_length, device, embedder):

        """ Load the data
        """

        data = read_csv(data)
        #data = data.head(1000)
        data['text'] = data['text'].apply(lambda x: PreProcess.text_preprocessing(x))

        target = data["target"]
        text = data["text"]

        text_embeddings = embedder.encode(text)

        x_train, x_test, y_train, y_test = train_test_split(
            text_embeddings, target, train_size=t_size, random_state=42,
        )

        return x_train, x_test, y_train, y_test


class IsBestModel(nn.Module):

    # define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, device):
        super().__init__()


        self.hidden_dim  = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=False)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.act = nn.Sigmoid()

    def forward(self, text):

        self.b_size = text.shape[1]
        hidden = self._init_hidden()

        packed_output, (hidden, cell) = self.lstm(text, hidden)
        hidden = cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        dense_outputs = self.fc(hidden)
        outputs = self.act(dense_outputs)

        return outputs

    def _init_hidden(self):
        return (torch.zeros(self.n_layers*2, self.b_size,  self.hidden_dim).to(self.device),
                torch.zeros(self.n_layers*2, self.b_size,  self.hidden_dim).to(self.device))


class IsBest:

    def __init__(self, batch_size_train, batch_size_val, epochs, maxlen, learning_rate, device):

        # self.filename = model_name
        # self.output_name = output_name
        self.device = device
        self.epochs = epochs
        self.maxlen = maxlen
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.learning_rate = learning_rate

        # Set device
        self.get_dev()


    def get_dev(self):
        """
        Get device information
        :return:
        """
        if self.device is not None:
            self.device = self.device
        else:
            self.device = 'cuda' if cuda.is_available() else 'cpu'

    def fit(self, x_train, x_test, y_train, y_test, output_name, batch_size_train, batch_size_val, epochs, learning_rate ):

        self.embedding_dim = x_train.shape[1]
        self.num_hidden_nodes = 32
        self.num_output_nodes = 1
        self.num_layers = 2
        self.bidirection = True
        self.dropout = 0.2
        self.output_name = output_name
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.epochs = epochs
        self.output_name = output_name
        self.learning_rate = learning_rate
        tensor_x = torch.as_tensor(x_train).to(self.device).type(torch.float64)#.long()
        tensor_y = torch.as_tensor(y_train.values).to(self.device).type(torch.float64)

        train_dataset = TensorDataset(tensor_x, tensor_y)

        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size_train)

        tensor_x = torch.as_tensor(x_test).to(self.device).type(torch.float64)
        tensor_y = torch.as_tensor(y_test.values).to(self.device).type(torch.float64)

        train_dataset = TensorDataset(tensor_x, tensor_y)
        self.test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size_val)

        self.model = IsBestModel( self.embedding_dim, self.embedding_dim,
                                  self.num_hidden_nodes, self.num_output_nodes, self.num_layers,
                           bidirectional=self.bidirection, dropout=self.dropout, device=self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()

        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        best_valid_loss = float('inf')

        for epoch in range(self.epochs):

            train_loss, train_acc = self.train()

            valid_loss, valid_acc = self.evaluate()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.output_name)

            print(f'\tTrain Loss: {train_loss:.3f} | Train AUC: {train_acc * 100:.2f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. AUC: {valid_acc * 100:.2f}')

    def train(self):

        epoch_loss = 0
        epoch_acc = 0

        self.model.train()

        for batch in self.train_dataloader:
            self.optimizer.zero_grad()

            batch_= batch[0].float().squeeze().unsqueeze(dim=0)
            predictions = self.model(batch_).squeeze()

            loss = self.criterion(predictions.float(), batch[1].float())

            acc = auroc(predictions.float(), batch[1].float())

            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(self.train_dataloader), epoch_acc / len(self.train_dataloader)

    def evaluate(self):

        epoch_loss = 0
        epoch_acc = 0

        self.model.eval()

        with torch.no_grad():
            for batch in self.train_dataloader:
                batch_ = batch[0].float().squeeze().unsqueeze(dim=0)
                predictions = self.model(batch_).squeeze()

                loss = self.criterion(predictions.float(), batch[1].float())
                acc = auroc(predictions.float(), batch[1].float())

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(self.train_dataloader), epoch_acc / len(self.train_dataloader)

    def load(self, model_name):
        self.embedding_dim = 768
        self.num_hidden_nodes = 32
        self.num_output_nodes = 1
        self.num_layers = 2
        self.bidirection = True
        self.dropout = 0.2
        self.model_name = model_name

        self.model = IsBestModel(self.embedding_dim, self.embedding_dim,
                                  self.num_hidden_nodes, self.num_output_nodes, self.num_layers,
                           bidirectional=True, dropout=self.dropout, device=self.device)

        self.model.load_state_dict(torch.load(args.model_name))\

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, embedded):

        prep_text = embedded.encode(text)

        #tensor = torch.as_tensor(prep_text).to(self.device).type(torch.float64).view(1, 1, -1)
        #with torch.no_grad():
            #return float(self.model(tensor.float()).cpu().detach().numpy())
        return float(self.model(torch.as_tensor(prep_text).to(self.device).view(1, 1, -1)).cpu().detach().numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-type', type=str,  default="inf") #both 'train'/'inf' | train or inference
    parser.add_argument('-batch_size_train', type=int,  default=1024, required=False) #train | batch_size for train
    parser.add_argument('-batch_size_val', type=int,  default=1024, required=False) #train | batch_size for validation
    parser.add_argument('-epochs', type=int,  default=20, required=False) #train  | num of epochs
    parser.add_argument('-maxlen', type=int, default=256, required=False) #train | max token len
    parser.add_argument('-output_name', type=str,  default="model.pth", required=False) #train | output name for saving model
    parser.add_argument('-learning_rate', type=str,  default=0.0001, required=False)  #train  | learning rate
    parser.add_argument('-device', type=str,  default=None, required=False)  #train   | device if none then if cuda is avaliable then cuda else cpu
    parser.add_argument('-dataset', type=str,  default="answers.csv", required=False) #train  | dataset

    parser.add_argument('-question', type=str,  default=" ", required=False) #inference  | text question
    parser.add_argument('-answer', type=str,  default=" ", required=False) #inference | text answer
    parser.add_argument('-model_name', type=str, default="model.pth", required=False) #inference/train | model name for loading

    args = parser.parse_args()

    embedder = Embedder(
        emberdder='DeepPavlov/rubert-base-cased',
        tokenizer='DeepPavlov/rubert-base-cased',
        device=PreProcess.get_dev(args.device),
        tokenizer_max_length=args.maxlen
    )
    classif = IsBest(args.batch_size_train, args.batch_size_val, args.epochs, args.maxlen, args.learning_rate,
                     args.device)

    if args.type == 'inf':
        # Just inference

        classif.load(args.model_name)

        a =  datetime.now()
        print(classif.predict(args.question + " " + args.answer, embedded=embedder))
        b = datetime.now()
        c = b -a
        print(c.seconds)
    else:

        x_train, x_test, y_train, y_test =  PreProcess.import_data(data=args.dataset, t_size=0.85,
                               max_length=args.maxlen, device=args.device, embedder=embedder)

        classif.fit(x_train, x_test, y_train, y_test,
                    args.output_name, args.batch_size_train,
                    args.batch_size_val, args.epochs, args.learning_rate)


