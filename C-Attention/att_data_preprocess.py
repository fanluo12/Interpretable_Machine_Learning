# #
# data pre-process functions
# #
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import torch
from torch.autograd import Variable

def get_label():
    data = pd.read_csv(os.path.join(os.path.join(Path(os.getcwd()).parent, "data"), "label.csv"))
    return data['label']

def get_dt():
    file_name = os.path.join(os.path.join(Path(os.getcwd()).parent, "data"), "pos.csv")
    return pd.read_csv(file_name)

def get_embed_dt():
    file_name = os.path.join(os.path.join(Path(os.getcwd()).parent, "data"), "embed.npy")
    print(file_name)
    return np.load(file_name)

def get_input_data():

    df = get_dt()

    POS_SET = ["NN", "NNS", "WRB", "MD", "JJR", "PRP", "VB", "IN", "$", "JJ", "RP", "PRP$", "CC", "CD", "VBG", "RBS", "PDT", "POS", "UH", "NNP", "TO", "DT", "RB","VBZ", "VBN", "WP", "VBP", "JJS", "VBD", "NNPS", "''", "EX", "RBR", "WP$", "FW", "WDT"]
    data = torch.tensor(np.array(df[POS_SET]))
    data = data.unsqueeze(1).type(torch.FloatTensor)
    df_train = data[:1163:]
    x_pos_text = data[1163:1292:]

    y_all = get_label()
    y_train = y_all[:1163].tolist()
    y_test = y_all[1163:1292].tolist()

    y_train = Variable(torch.tensor(y_train).type(torch.LongTensor))
    y_test = Variable(torch.tensor(y_test).type(torch.LongTensor))

    embed = get_embed_dt()
    len_sentences = len(embed[0])
    embed_all = torch.from_numpy(embed)

    embed_train = embed_all[:1163, :, :]
    embed_test = embed_all[1163:1292, :, :]
    x_embed_test = embed_test

    x_pos_train, x_pos_valid, x_embed_train, x_embed_valid, y_train, y_valid = train_test_split(df_train, embed_train, y_train, test_size=0.1)

    return len_sentences, x_pos_train, x_embed_train, y_train, x_pos_text, x_embed_test, y_test, x_pos_valid, x_embed_valid, y_valid
