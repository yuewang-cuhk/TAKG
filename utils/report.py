import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np


def export_train_and_valid_reward(train_reward, valid_reward, plot_every, path):
    # Export the results to a csv file
    labels = ['Training reward:,', 'Validation reward:,']
    float_lists = [train_reward, valid_reward]
    with open(path + '.csv', 'w') as result_csv:
        for i in range(len(labels)):
            result_csv.write(labels[i] + concat_float_list(float_lists[i], ',') + '\n')
    # Export the plots to pdf file
    plot_train_valid_curve(train_reward, valid_reward, plot_every, path, 'Reward')


def export_train_and_valid_loss(train_loss, valid_loss, train_ppl, valid_ppl, plot_every, path):
    """
    :param train_loss: a list of float
    :param valid_loss: a list of float
    :param train_ppl: a list of float
    :param valid_ppl: a list of float
    :param plot_every: int
    :param path: str
    :return:
    """
    # Export the results to a csv file
    labels = ['Training loss:,', 'Validation loss:,', 'Training perplexity:,', 'Validation Perplexity:,']
    float_lists = [train_loss, valid_loss, train_ppl, valid_ppl]
    with open(path + '.csv', 'w') as result_csv:
        for i in range(len(labels)):
            result_csv.write(labels[i] + concat_float_list(float_lists[i], ',') + '\n')
    # Export the plots to pdf file
    plot_train_valid_curve(train_loss, valid_loss, plot_every, path, 'Loss')
    plot_train_valid_curve(train_ppl, valid_ppl, plot_every, path, 'Perplexity')


def concat_float_list(list, delimiter=','):
    return delimiter.join([str(l) for l in list])

def plot_train_valid_curve(train_loss, valid_loss, plot_every, path, loss_label):
    #plt.ioff()
    title = "Training and validation %s for every %d iterations" % (loss_label.lower(), plot_every)
    plt.figure()
    plt.title(title)
    plt.xlabel("Checkpoints")
    plt.ylabel(loss_label)
    num_checkpoints = len(train_loss)
    X = list(range(num_checkpoints))
    plt.plot(X, train_loss, label="training")
    plt.plot(X, valid_loss, label="validation")
    plt.legend()
    plt.savefig("%s_%s.pdf" % (path, loss_label.lower()))

if __name__ == '__main__':
    train_loss = [20.1,15.3,12.3,11.0,10.0]
    valid_loss = [30.2,29.2,25.2,21.3,20.2]
    train_ppl = [10.1,5.3,2.3,1.0,1.0]
    valid_ppl = [20.2,19.2,15.2,11.3,10.2]

    plot_every = 4000
    path = '../exp/debug/valid_train_curve'
    export_train_and_valid_loss(train_loss, valid_loss, train_ppl, valid_ppl, plot_every, path)
