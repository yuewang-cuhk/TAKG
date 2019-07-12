import torch
import logging
import time
import sys
import argparse

import config
from sequence_generator import SequenceGenerator
from utils.time_log import time_since
from evaluate import evaluate_beam_search
from utils.data_loader import load_data_and_vocab
import pykp.io
from pykp.model import Seq2SeqModel, NTM

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def init_pretrained_model(opt):
    model = Seq2SeqModel(opt)
    model.load_state_dict(torch.load(opt.model))
    model.to(opt.device)
    model.eval()
    if opt.use_topic_represent:
        ntm_model = NTM(opt)
        ntm_model.load_state_dict(torch.load(opt.ntm_model))
        ntm_model.to(opt.device)
        ntm_model.eval()
    else:
        ntm_model = None
    return model, ntm_model


def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")

    assert opt.model.count('/') == 2 and all([tag in opt.model for tag in ['vs', 'emb', 'dec', 'model']])
    opt.vocab_size = [int(w.replace('vs', '')) for w in opt.model.split('.') if w.startswith('vs')][0]
    opt.word_vec_size = [int(w.replace('emb', '')) for w in opt.model.split('.') if w.startswith('emb')][0]
    opt.decoder_size = [int(w.replace('dec', '')) for w in opt.model.split('.') if w.startswith('dec')][0]
    opt.encoder_size = int(opt.decoder_size / 2)

    opt.data_tag = opt.model.split('/')[1].split('.')[0]
    opt.data = "processed_data/{}/".format(opt.data_tag)
    opt.vocab = opt.data
    opt.exp = 'predict__' + '__'.join(opt.model.split('/')[1:])

    opt.replace_unk = True

    if opt.trial:
        opt.exp = 'trial.' + opt.exp
        opt.batch_size = 8

    if ".topic_num" in opt.model:
        opt.topic_num = int([t.replace('topic_num', '') for t in opt.model.split('.') if 'topic_num' in t][0])

    if ".copy" in opt.model:
        opt.copy_attention = True

    if ".topic_copy" in opt.model:
        opt.topic_copy = True

    if ".no_topic_dec" in opt.model:
        opt.topic_dec = False
    elif ".use_topic" in opt.model:
        opt.topic_dec = True

    if ".topic_attn." in opt.model:
        opt.topic_attn = True

    if ".topic_attn_in." in opt.model:
        opt.topic_attn_in = True

    if ".z_topic" in opt.model:
        opt.topic_type = 'z'

    if ".use_topic" in opt.model:
        opt.use_topic_represent = True
        if '.joint_train' in opt.model:
            opt.ntm_model = opt.model.replace('model-', 'model_ntm-')
        assert os.path.exists(opt.ntm_model), 'please specify the ntm model'

    if opt.n_best < 0:
        opt.n_best = opt.beam_size

    # fill time into the name
    if opt.pred_path.find('%s') > 0:
        opt.pred_path = opt.pred_path % opt.exp

    if not os.path.exists(opt.pred_path):
        os.makedirs(opt.pred_path)

    return opt


def predict(test_data_loader, model, ntm_model, opt):
    if opt.delimiter_type == 0:
        delimiter_word = pykp.io.SEP_WORD
    else:
        delimiter_word = pykp.io.EOS_WORD
    generator = SequenceGenerator(model,
                                  ntm_model,
                                  opt.use_topic_represent,
                                  opt.topic_type,
                                  bos_idx=opt.word2idx[pykp.io.BOS_WORD],
                                  eos_idx=opt.word2idx[pykp.io.EOS_WORD],
                                  pad_idx=opt.word2idx[pykp.io.PAD_WORD],
                                  beam_size=opt.beam_size,
                                  max_sequence_length=opt.max_length,
                                  copy_attn=opt.copy_attention,
                                  coverage_attn=opt.coverage_attn,
                                  review_attn=opt.review_attn,
                                  length_penalty_factor=opt.length_penalty_factor,
                                  coverage_penalty_factor=opt.coverage_penalty_factor,
                                  length_penalty=opt.length_penalty,
                                  coverage_penalty=opt.coverage_penalty,
                                  cuda=opt.gpuid > -1,
                                  n_best=opt.n_best,
                                  block_ngram_repeat=opt.block_ngram_repeat,
                                  ignore_when_blocking=opt.ignore_when_blocking
                                  )

    evaluate_beam_search(generator, test_data_loader, opt, delimiter_word)


def main(opt):
    try:
        start_time = time.time()
        load_data_time = time_since(start_time)
        test_data_loader, word2idx, idx2word, vocab, bow_dictionary = load_data_and_vocab(opt, load_train=False)
        opt.bow_vocab_size = len(bow_dictionary)
        model, ntm_model = init_pretrained_model(opt)
        logging.info('Time for loading the data and model: %.1f' % load_data_time)
        start_time = time.time()

        predict(test_data_loader, model, ntm_model, opt)

        total_testing_time = time_since(start_time)
        logging.info('Time for a complete testing: %.1f' % total_testing_time)
        print('Time for a complete testing: %.1f' % total_testing_time)
        sys.stdout.flush()
    except Exception as e:
        logging.exception("message")
    return

    pass


if __name__ == '__main__':
    # load settings for predicting
    parser = argparse.ArgumentParser(
        description='predict.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.model_opts(parser)
    config.my_own_opts(parser)
    config.predict_opts(parser)
    config.vocab_opts(parser)

    opt = parser.parse_args()
    opt = process_opt(opt)

    logging = config.init_logging(log_file=opt.pred_path + '/output.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
