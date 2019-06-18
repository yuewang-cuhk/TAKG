import torch
import argparse
import logging
import json
import time
import numpy as np
import random
from itertools import chain
from torch.optim import Adam

import config
import train_mixture
from pykp.io import SEP_WORD, EOS_WORD
from pykp.model import Seq2SeqModel, NTM

from utils.time_log import time_since
from utils.data_loader import load_data_and_vocab

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if torch.cuda.is_available() and not opt.gpuid:
        opt.gpuid = 0

    if opt.delimiter_type == 0:
        opt.delimiter_word = SEP_WORD
    else:
        opt.delimiter_word = EOS_WORD

    # my configuration
    opt.data = "processed_data/{}/".format(opt.data_tag)
    opt.vocab = opt.data
    opt.exp = 'trial.' + opt.data_tag if opt.trial else opt.data_tag

    # seq2seq setting
    if 'Weibo' in opt.data_tag:
        opt.vocab_size = 50000
        opt.word_vec_size = 150
    elif 'Twitter' in opt.data_tag:
        opt.vocab_size = 30000
        opt.word_vec_size = 150
    elif 'StackExchange' in opt.data_tag:
        opt.vocab_size = 50000
        opt.word_vec_size = 150
    else:
        print('Wrong data_tag!!')
        return
    opt.encoder_size = 150
    opt.decoder_size = 300
    size_tag = ".emb{}".format(opt.word_vec_size) + ".vs{}".format(opt.vocab_size) + ".dec{}".format(opt.decoder_size)

    # only train ntm
    if opt.only_train_ntm:
        assert opt.ntm_warm_up_epochs > 0 and not opt.load_pretrain_ntm
        opt.exp += '.topic_num{}'.format(opt.topic_num)
        opt.exp += '.ntm_warm_up_%d' % opt.ntm_warm_up_epochs
        opt.model_path = opt.model_path % (opt.exp, opt.timemark)
        if not os.path.exists(opt.model_path):
            os.makedirs(opt.model_path)
        print("Only training the ntm for %d epochs and save it to %s" % (opt.ntm_warm_up_epochs, opt.model_path))
        return opt

    # joint train settings
    if opt.joint_train:
        opt.exp += '.joint_train'
        if opt.add_two_loss:
            opt.exp += '.add_two_loss'
        if opt.joint_train_strategy != 'p_1_joint':
            opt.exp += '.' + opt.joint_train_strategy
            opt.p_seq2seq_e = int(opt.joint_train_strategy.split('_')[1])
            if opt.joint_train_strategy.split('_')[-1] != 'joint':
                opt.iterate_train_ntm = True

    # adding topic settings
    if opt.use_topic_represent:
        opt.exp += '.use_topic'
        opt.exp += '.topic_num{}'.format(opt.topic_num)

        if opt.topic_type == 'z':
            opt.exp += '.z_topic'

        if opt.topic_attn:
            opt.exp += '.topic_attn'

        if not opt.topic_dec:
            opt.exp += '.no_topic_dec'

        if opt.topic_copy:
            opt.exp += '.topic_copy'

        if opt.topic_attn_in:
            opt.exp += '.topic_attn_in'

        if opt.load_pretrain_ntm:
            has_topic_num = [t for t in opt.check_pt_ntm_model_path.split('.') if 'topic_num' in t]
            if len(has_topic_num) != 0:
                assert opt.topic_num == int(has_topic_num[0].replace('topic_num', ''))

            ntm_tag = '.'.join(opt.check_pt_ntm_model_path.split('/')[-1].split('.')[:-1])
            # opt.exp += '.ntm_%s' % ntm_tag
        else:
            opt.exp += '.ntm_warm_up_%d' % opt.ntm_warm_up_epochs

    if opt.bridge != "copy":
        opt.exp += '.{}_bridge'.format(opt.bridge)

    if opt.copy_attention:
        opt.exp += '.copy'

    opt.exp += '.seed{}'.format(opt.seed)
    opt.exp += size_tag

    # fill time into the name
    if opt.model_path.find('%s') > 0:
        opt.model_path = opt.model_path % (opt.exp, opt.timemark)

    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)

    logging.info('Model_PATH : ' + opt.model_path)

    # dump the setting (opt) to disk in order to reuse easily
    if opt.train_from:
        opt = torch.load(
            open(os.path.join(opt.model_path, 'initial.config'), 'rb')
        )
    else:
        torch.save(opt,
                   open(os.path.join(opt.model_path, 'initial.config'), 'wb')
                   )
        json.dump(vars(opt), open(os.path.join(opt.model_path, 'initial.json'), 'w'))

    return opt


def init_optimizers(model, ntm_model, opt):
    optimizer_seq2seq = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    optimizer_ntm = Adam(params=filter(lambda p: p.requires_grad, ntm_model.parameters()), lr=opt.learning_rate)
    whole_params = chain(model.parameters(), ntm_model.parameters())
    optimizer_whole = Adam(params=filter(lambda p: p.requires_grad, whole_params), lr=opt.learning_rate)

    return optimizer_seq2seq, optimizer_ntm, optimizer_whole


def main(opt):
    try:
        start_time = time.time()
        train_data_loader, train_bow_loader, valid_data_loader, valid_bow_loader, \
        word2idx, idx2word, vocab, bow_dictionary = load_data_and_vocab(opt, load_train=True)
        opt.bow_vocab_size = len(bow_dictionary)
        load_data_time = time_since(start_time)
        logging.info('Time for loading the data: %.1f' % load_data_time)

        start_time = time.time()
        model = Seq2SeqModel(opt).to(opt.device)
        ntm_model = NTM(opt).to(opt.device)
        optimizer_seq2seq, optimizer_ntm, optimizer_whole = init_optimizers(model, ntm_model, opt)

        train_mixture.train_model(model,
                                  ntm_model,
                                  optimizer_seq2seq,
                                  optimizer_ntm,
                                  optimizer_whole,
                                  train_data_loader,
                                  valid_data_loader,
                                  bow_dictionary,
                                  train_bow_loader,
                                  valid_bow_loader,
                                  opt)

        training_time = time_since(start_time)

        logging.info('Time for training: %.1f' % training_time)

    except Exception as e:
        logging.exception("message")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.my_own_opts(parser)
    config.vocab_opts(parser)
    config.model_opts(parser)
    config.train_opts(parser)
    opt = parser.parse_args()
    opt = process_opt(opt)
    opt.input_feeding = False
    opt.copy_input_feeding = False

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")

    logging = config.init_logging(log_file=opt.model_path + '/output.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
