# from nltk.stem.porter import *
import torch
# from utils import Progbar
# from pykp.metric.bleu import bleu
from pykp.masked_loss import masked_cross_entropy
from utils.statistics import LossStatistics, RewardStatistics
import time
from utils.time_log import time_since
# from nltk.stem.porter import *
import pykp
import logging
import numpy as np
from collections import defaultdict
import os
import sys
from utils.string_helper import *
from torch.nn import functional as F


def evaluate_loss(data_loader, model, ntm_model, opt):
    model.eval()
    ntm_model.eval()
    evaluation_loss_sum = 0.0
    total_trg_tokens = 0
    n_batch = 0
    loss_compute_time_total = 0.0
    forward_time_total = 0.0
    print("Evaluate loss for %d batches" % len(data_loader))
    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            if not opt.one2many:  # load one2one dataset
                src, src_lens, src_mask, trg, trg_lens, trg_mask, src_oov, trg_oov, oov_lists, src_bow = batch
            else:  # load one2many dataset
                src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _ = batch
                num_trgs = [len(trg_str_list) for trg_str_list in
                            trg_str_2dlist]  # a list of num of targets in each batch, with len=batch_size

            max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

            batch_size = src.size(0)
            n_batch += batch_size

            # move data to GPU if available
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            trg = trg.to(opt.device)
            trg_mask = trg_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            trg_oov = trg_oov.to(opt.device)

            if opt.use_topic_represent:
                src_bow = src_bow.to(opt.device)
                src_bow_norm = F.normalize(src_bow)
                if opt.topic_type == 'z':
                    topic_represent, _, _, _, _ = ntm_model(src_bow_norm)
                else:
                    _, topic_represent, _, _, _ = ntm_model(src_bow_norm)
            else:
                topic_represent = None

            start_time = time.time()

            # one2one setting
            decoder_dist, h_t, attention_dist, encoder_final_state, coverage, _, _, _ \
                = model(src, src_lens, trg, src_oov, max_num_oov, src_mask, topic_represent)

            forward_time = time_since(start_time)
            forward_time_total += forward_time

            start_time = time.time()
            if opt.copy_attention:  # Compute the loss using target with oov words
                loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                                            opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage,
                                            coverage_loss=False)
            else:  # Compute the loss using target without oov words
                loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                            opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage,
                                            coverage_loss=False)
            loss_compute_time = time_since(start_time)
            loss_compute_time_total += loss_compute_time

            evaluation_loss_sum += loss.item()
            total_trg_tokens += sum(trg_lens)

            if (batch_i + 1) % (len(data_loader) // 5) == 0:
                print("Train: %d/%d batches, current avg loss: %.3f" %
                      ((batch_i + 1), len(data_loader), evaluation_loss_sum / total_trg_tokens))

    eval_loss_stat = LossStatistics(evaluation_loss_sum, total_trg_tokens, n_batch, forward_time=forward_time_total,
                                    loss_compute_time=loss_compute_time_total)
    return eval_loss_stat


def preprocess_beam_search_result(beam_search_result, idx2word, vocab_size, oov_lists, eos_idx, unk_idx, replace_unk,
                                  src_str_list):
    batch_size = beam_search_result['batch_size']
    predictions = beam_search_result['predictions']
    scores = beam_search_result['scores']
    attention = beam_search_result['attention']
    assert len(predictions) == batch_size
    pred_list = []  # a list of dict, with len = batch_size
    for pred_n_best, score_n_best, attn_n_best, oov, src_word_list in zip(predictions, scores, attention, oov_lists,
                                                                          src_str_list):
        # attn_n_best: list of tensor with size [trg_len, src_len], len=n_best
        pred_dict = {}
        sentences_n_best = []
        for pred, attn in zip(pred_n_best, attn_n_best):
            sentence = prediction_to_sentence(pred, idx2word, vocab_size, oov, eos_idx, unk_idx, replace_unk,
                                              src_word_list, attn)
            # sentence = [idx2word[int(idx.item())] if int(idx.item()) < vocab_size else oov[int(idx.item())-vocab_size] for idx in pred[:-1]]
            sentences_n_best.append(sentence)
        pred_dict[
            'sentences'] = sentences_n_best  # a list of list of word, with len [n_best, out_seq_len], does not include tbe final <EOS>
        pred_dict['scores'] = score_n_best  # a list of zero dim tensor, with len [n_best]
        pred_dict[
            'attention'] = attn_n_best  # a list of FloatTensor[output sequence length, src_len], with len = [n_best]
        pred_list.append(pred_dict)
    return pred_list


def evaluate_beam_search(generator, one2many_data_loader, opt, delimiter_word='<sep>'):
    # score_dict_all = defaultdict(list)  # {'precision@5':[],'recall@5':[],'f1_score@5':[],'num_matches@5':[],'precision@10':[],'recall@10':[],'f1score@10':[],'num_matches@10':[]}
    # file for storing the predicted keyphrases
    pred_fn = os.path.join(opt.pred_path, 'predictions.txt')
    pred_output_file = open(pred_fn, "w")
    # debug
    interval = 100

    with torch.no_grad():
        start_time = time.time()
        print("Receiving %d batches with batch_size=%d" % (len(one2many_data_loader), opt.batch_size))
        for batch_i, batch in enumerate(one2many_data_loader):
            if (batch_i + 1) % interval == 0:
                print("Batch %d: Time for running beam search: %.1f" % (batch_i + 1, time_since(start_time)))
                sys.stdout.flush()
            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, _, _, _, _, original_idx_list, src_bow = batch
            """
            src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
            src_lens: a list containing the length of src sequences for each batch, with len=batch
            src_mask: a FloatTensor, [batch, src_seq_len]
            src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
            oov_lists: a list of oov words for each src, 2dlist
            """
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            src_bow = src_bow.to(opt.device)

            beam_search_result = generator.beam_search(src, src_lens, src_oov, src_mask, src_bow, oov_lists,
                                                       opt.word2idx, opt.max_eos_per_output_seq)
            pred_list = preprocess_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size, oov_lists,
                                                      opt.word2idx[pykp.io.EOS_WORD], opt.word2idx[pykp.io.UNK_WORD],
                                                      opt.replace_unk, src_str_list)
            # list of {"sentences": [], "scores": [], "attention": []}

            # recover the original order in the dataset
            seq_pairs = sorted(zip(original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists),
                               key=lambda p: p[0])
            original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists = zip(*seq_pairs)

            # Process every src in the batch
            for src_str, trg_str_list, pred, oov in zip(src_str_list, trg_str_2dlist, pred_list, oov_lists):
                # src_str: a list of words; trg_str: a list of keyphrases, each keyphrase is a list of words
                # pred_seq_list: a list of sequence objects, sorted by scores
                # oov: a list of oov words
                pred_str_list = pred[
                    'sentences']  # predicted sentences from a single src, a list of list of word, with len=[beam_size, out_seq_len], does not include the final <EOS>
                pred_score_list = pred['scores']
                pred_attn_list = pred[
                    'attention']  # a list of FloatTensor[output sequence length, src_len], with len = [n_best]

                if opt.one2many:
                    all_keyphrase_list = []  # a list of word list contains all the keyphrases in the top max_n sequences decoded by beam search
                    for word_list in pred_str_list:
                        all_keyphrase_list += split_concated_keyphrases(word_list, delimiter_word)
                        # not_duplicate_mask = check_duplicate_keyphrases(all_keyphrase_list)
                    # pred_str_list = [word_list for word_list, is_keep in zip(all_keyphrase_list, not_duplicate_mask) if is_keep]
                    pred_str_list = all_keyphrase_list

                # output the predicted keyphrases to a file
                pred_print_out = ''
                for word_list_i, word_list in enumerate(pred_str_list):
                    if word_list_i < len(pred_str_list) - 1:
                        pred_print_out += '%s;' % ' '.join(word_list)
                    else:
                        pred_print_out += '%s' % ' '.join(word_list)
                pred_print_out += '\n'
                pred_output_file.write(pred_print_out)

    pred_output_file.close()
    logging.info("Writing to %s" % pred_fn)
    logging.info("done!")
