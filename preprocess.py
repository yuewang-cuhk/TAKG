import argparse
from collections import Counter
import torch
import pykp.io
import config
import gensim
import os
import time
import re


def read_src_trg_files(opt, tag="train"):
    '''
    Read data according to the tag (train/valid/test), return a list of (src, trg) pairs
    '''
    if tag == "train":
        src_file = opt.train_src
        trg_file = opt.train_trg
    elif tag == "valid":
        src_file = opt.valid_src
        trg_file = opt.valid_trg
    else:
        src_file = opt.test_src
        trg_file = opt.test_trg

    tokenized_src = []
    tokenized_trg = []

    for src_line, trg_line in zip(open(src_file, 'r'), open(trg_file, 'r')):
        # process src and trg line
        src_word_list = src_line.strip().split(' ')
        trg_list = trg_line.strip().split(';')  # a list of target sequences
        trg_word_list = [trg.strip().split(' ') for trg in trg_list]

        # Truncate the sequence if it is too long
        src_word_list = src_word_list[:opt.max_src_len]
        if tag != "test":
            trg_word_list = [trg_list[:opt.max_trg_len] for trg_list in trg_word_list]

        # Append the lines to the data
        tokenized_src.append(src_word_list)
        tokenized_trg.append(trg_word_list)

    assert len(tokenized_src) == len(tokenized_trg), \
        'the number of records in source and target are not the same'

    tokenized_pairs = list(zip(tokenized_src, tokenized_trg))
    print("Finish reading %d lines of data from %s and %s" % (len(tokenized_src), src_file, trg_file))
    return tokenized_pairs


def build_vocab(tokenized_src_trg_pairs):
    '''
    Build the vocabulary from the training (src, trg) pairs
    :param tokenized_src_trg_pairs: list of (src, trg) pairs
    :return: word2idx, idx2word, token_freq_counter
    '''
    token_freq_counter = Counter()
    for src_word_list, trg_word_lists in tokenized_src_trg_pairs:
        token_freq_counter.update(src_word_list)
        for word_list in trg_word_lists:
            token_freq_counter.update(word_list)

    # Discard special tokens if already present
    special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>', '<sep>']
    num_special_tokens = len(special_tokens)

    for s_t in special_tokens:
        if s_t in token_freq_counter:
            del token_freq_counter[s_t]

    word2idx = dict()
    idx2word = dict()
    for idx, word in enumerate(special_tokens):
        # '<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3
        word2idx[word] = idx
        idx2word[idx] = word

    sorted_word2idx = sorted(token_freq_counter.items(), key=lambda x: x[1], reverse=True)

    sorted_words = [x[0] for x in sorted_word2idx]

    for idx, word in enumerate(sorted_words):
        word2idx[word] = idx + num_special_tokens

    for idx, word in enumerate(sorted_words):
        idx2word[idx + num_special_tokens] = word

    return word2idx, idx2word, token_freq_counter


def make_bow_dictionary(tokenized_src_trg_pairs, data_dir, bow_vocab):
    '''
    Build bag-of-word dictionary from tokenized_src_trg_pairs
    :param tokenized_src_trg_pairs: a list of (src, trg) pairs
    :param data_dir: data address, for distinguishing Weibo/Twitter/StackExchange
    :param bow_vocab: the size the bow vocabulary
    :return: bow_dictionary, a gensim.corpora.Dictionary object
    '''
    doc_bow = []
    tgt_set = set()

    for src, tgt in tokenized_src_trg_pairs:
        cur_bow = []
        cur_bow.extend(src)
        for t in tgt:
            cur_bow.extend(t)
            [tgt_set.add(x) for x in t]
        # Remove token that does not contain letters
        if 'Weibo' not in data_dir:
            cur_bow = list(filter(lambda x: re.search('[a-zA-Z]', x), cur_bow))
        doc_bow.append(cur_bow)

    bow_dictionary = gensim.corpora.Dictionary(doc_bow)
    # Remove single letter or character tokens
    len_1_words = list(filter(lambda w: len(w) == 1, bow_dictionary.values()))
    bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, len_1_words)))

    def read_stopwords(fn):
        return set([line.strip() for line in open(fn, encoding='utf-8') if len(line.strip()) != 0])

    # Read stopwords from file (bow vocabulary should not contain stopwords)
    if 'Weibo' in data_dir:
        stopwords1 = read_stopwords("stopwords/stopwords.zh.txt")
        stopwords2 = read_stopwords("stopwords/stopwords.weibo.txt")
        final_stopwords = stopwords1.union(stopwords2)
    elif 'Twitter' in data_dir:
        STOPWORDS = gensim.parsing.preprocessing.STOPWORDS
        stopwords1 = read_stopwords("stopwords/stopwords.en.txt")
        stopwords2 = read_stopwords("stopwords/stopwords.twitter.txt")
        final_stopwords = set(STOPWORDS).union(stopwords1).union(stopwords2)
    elif 'StackExchange' in data_dir:
        STOPWORDS = gensim.parsing.preprocessing.STOPWORDS
        stopwords1 = read_stopwords("stopwords/stopwords.en.txt")
        stopwords2 = read_stopwords("stopwords/stopwords.SE.txt")
        final_stopwords = set(STOPWORDS).union(stopwords1).union(stopwords2)

    bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, final_stopwords)))

    print("The original bow vocabulary: %d" % len(bow_dictionary))
    bow_dictionary.filter_extremes(no_below=3, keep_n=bow_vocab)
    bow_dictionary.compactify()
    bow_dictionary.id2token = dict([(id, t) for t, id in bow_dictionary.token2id.items()])
    # for debug
    sorted_dfs = sorted(bow_dictionary.dfs.items(), key=lambda x: x[1], reverse=True)
    sorted_dfs_token = [(bow_dictionary.id2token[id], cnt) for id, cnt in sorted_dfs]
    print('The top 50 non-stop-words: ', sorted_dfs_token[:50])
    return bow_dictionary


def main(opt):
    t0 = time.time()
    # Tokenize training data, return a list of tuple, (src_word_list, [trg_1_word_list, trg_2_word_list, ...])
    tokenized_train_pairs = read_src_trg_files(opt, "train")

    # Build vocabulary from training src and trg
    print("Building vocabulary from training data")
    word2idx, idx2word, token_freq_counter = build_vocab(tokenized_train_pairs)
    print("Total vocab_size: %d, predefined vocab_size: %d" % (len(word2idx), opt.vocab_size))

    # Build bag-of-word dictionary from training data
    print("Building bow dictionary from training data")
    bow_dictionary = make_bow_dictionary(tokenized_train_pairs, opt.data_dir, opt.bow_vocab)
    print("Bow dict_size: %d after filtered" % len(bow_dictionary))

    print("Dumping dict to disk: %s\n" % (opt.res_data_dir + '/vocab.pt'))
    torch.save([word2idx, idx2word, token_freq_counter, bow_dictionary],
               open(opt.res_data_dir + '/vocab.pt', 'wb'))

    # Build training set for one2one training mode
    # train_one2one is a list of dict, with fields src, trg, src_oov, oov_dict, oov_list, etc.
    train_one2one = pykp.io.build_dataset(
        tokenized_train_pairs, word2idx, bow_dictionary, opt, mode='one2one')
    print("Dumping train one2one to disk: %s\n" % (opt.res_data_dir + '/train.one2one.pt'))
    torch.save(train_one2one, open(opt.res_data_dir + '/train.one2one.pt', 'wb'))

    # Processing valid dataset
    tokenized_valid_pairs = read_src_trg_files(opt, "valid")
    valid_one2one = pykp.io.build_dataset(
        tokenized_valid_pairs, word2idx, bow_dictionary, opt, mode='one2one')

    print("Dumping valid to disk: %s\n" % (opt.res_data_dir + '/valid.ne2one.pt'))
    torch.save(valid_one2one, open(opt.res_data_dir + '/valid.one2one.pt', 'wb'))

    # Processing test dataset
    tokenized_test_pairs = read_src_trg_files(opt, "test")
    # Build test set for one2many training mode
    test_one2many = pykp.io.build_dataset(
        tokenized_test_pairs, word2idx, bow_dictionary, opt, mode='one2many')

    print("Dumping test to disk: %s\n" % (opt.res_data_dir + '/test.one2many.pt'))
    torch.save(test_one2many, open(opt.res_data_dir + '/test.one2many.pt', 'wb'))

    print('#pairs of train_one2one  = %d' % len(train_one2one))
    print('#pairs of valid_one2one  = %d' % len(valid_one2one))
    print('#pairs of test_one2many  = %d' % len(test_one2many))

    print('\nFinish and take %.2f seconds' % (time.time() - t0))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess_conv_bow.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    config.vocab_opts(parser)
    opt = parser.parse_args()
    opt.train_src = opt.data_dir + '/train_src.txt'
    opt.train_trg = opt.data_dir + '/train_trg.txt'
    opt.valid_src = opt.data_dir + '/valid_src.txt'
    opt.valid_trg = opt.data_dir + '/valid_trg.txt'
    opt.test_src = opt.data_dir + '/test_src.txt'
    opt.test_trg = opt.data_dir + '/test_trg.txt'

    if 'Twitter' in opt.data_dir:
        opt.vocab_size = 30000
    elif 'StackExchange' in opt.data_dir:
        opt.max_src_len = 150

    data_fn = opt.data_dir.rstrip('/').split('/')[-1] + '_s{}_t{}'.format(opt.max_src_len, opt.max_trg_len)

    opt.res_data_dir = "processed_data/%s" % data_fn
    if not os.path.exists(opt.res_data_dir):
        os.mkdir(opt.res_data_dir)

    main(opt)
