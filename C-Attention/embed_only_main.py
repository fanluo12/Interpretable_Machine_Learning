########################### import ###############################

from embed_only_train import *
from att_test import *
from torch.utils.data import *
from att_data_preprocess import *

def main():
    batch_size = 50
    EPOCH = 1
    len_sentences, x_pos_train, x_embed_train, y_train, x_pos_test, x_embed_test, y_test, x_pos_valid, x_embed_valid, y_valid = get_input_data()
    train_data = []
    for i in range(len(x_embed_train)):
        train_data.append([x_embed_train[i], y_train[i]])
    val_data = []
    for i in range(len(x_embed_valid)):
        val_data.append([x_embed_valid[i], y_valid[i]])

    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_data = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)


    best_model = train(EPOCH, len_sentences, train_data, val_data)

    pre_np = test(best_model,  x_embed_test)
    single_gen_accu(pre_np, y_test)


if __name__ == '__main__':
    main()