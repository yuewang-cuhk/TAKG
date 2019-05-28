# -*- coding: utf-8 -*-
"""
Python File Template
Built on the source code of seq2seq-keyphrase-pytorch: https://github.com/memray/seq2seq-keyphrase-pytorch
"""
import numpy as np
import torch
import torch.utils.data

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
SEP_WORD = '<sep>'
DIGIT = '<digit>'


class KeyphraseDataset(torch.utils.data.Dataset):
    def __init__(self, examples, word2idx, idx2word, bow_dictionary,
                 type='one2one', delimiter_type=0, load_train=True, remove_src_eos=False):
        # keys of matter. `src_oov` is for mapping pointed word to dict,
        # `oov_dict` is for determining the dim of predicted logit: dim=vocab_size+max_oov_dict_in_batch
        assert type in ['one2one', 'one2many']
        keys = ['src', 'trg', 'trg_copy', 'src_oov', 'oov_dict', 'oov_list', 'src_str', 'trg_str', 'src_bow']

        filtered_examples = []

        for e in examples:
            filtered_example = {}
            for k in keys:
                filtered_example[k] = e[k]
            if 'oov_list' in filtered_example:
                filtered_example['oov_number'] = len(filtered_example['oov_list'])

            filtered_examples.append(filtered_example)

        self.examples = filtered_examples
        self.word2idx = word2idx
        self.id2xword = idx2word
        self.bow_dictionary = bow_dictionary
        self.pad_idx = word2idx[PAD_WORD]
        self.type = type
        if delimiter_type == 0:
            self.delimiter = self.word2idx[SEP_WORD]
        else:
            self.delimiter = self.word2idx[EOS_WORD]
        self.load_train = load_train
        self.remove_src_eos = remove_src_eos

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def _pad(self, input_list):
        input_list_lens = [len(l) for l in input_list]
        max_seq_len = max(input_list_lens)
        padded_batch = self.pad_idx * np.ones((len(input_list), max_seq_len))

        for j in range(len(input_list)):
            current_len = input_list_lens[j]
            padded_batch[j][:current_len] = input_list[j]

        padded_batch = torch.LongTensor(padded_batch)

        input_mask = torch.ne(padded_batch, self.pad_idx)
        input_mask = input_mask.type(torch.FloatTensor)

        return padded_batch, input_list_lens, input_mask

    def _pad_bow(self, input_list):
        bow_vocab = len(self.bow_dictionary)
        res_src_bow = np.zeros((len(input_list), bow_vocab))
        for idx, bow in enumerate(input_list):
            bow_k = [k for k, v in bow]
            bow_v = [v for k, v in bow]
            res_src_bow[idx, bow_k] = bow_v
        return torch.FloatTensor(res_src_bow)

    def collate_bow(self, batches):
        src_bow = [b['src_bow'] for b in batches]
        return self._pad_bow(src_bow)

    def collate_fn_one2one(self, batches):
        '''
        Puts each data field into a tensor with outer dimension batch size"
        '''
        assert self.type == 'one2one', 'The type of dataset should be one2one.'
        if self.remove_src_eos:
            # source with oov words replaced by <unk>
            src = [b['src'] for b in batches]
            # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
            src_oov = [b['src_oov'] for b in batches]
        else:
            # source with oov words replaced by <unk>
            src = [b['src'] + [self.word2idx[EOS_WORD]] for b in batches]
            # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
            src_oov = [b['src_oov'] + [self.word2idx[EOS_WORD]] for b in batches]

        # target_input: input to decoder, ends with <eos> and oovs are replaced with <unk>
        trg = [b['trg'] + [self.word2idx[EOS_WORD]] for b in batches]

        # target for copy model, ends with <eos>, oovs are replaced with temporary idx, e.g. 50000, 50001 etc.)
        trg_oov = [b['trg_copy'] + [self.word2idx[EOS_WORD]] for b in batches]

        oov_lists = [b['oov_list'] for b in batches]
        src_bow = [b['src_bow'] for b in batches]

        # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
        seq_pairs = sorted(zip(src, trg, trg_oov, src_oov, oov_lists, src_bow), key=lambda p: len(p[0]), reverse=True)
        src, trg, trg_oov, src_oov, oov_lists, src_bow = zip(*seq_pairs)

        # pad the src and target sequences with <pad> token and convert to LongTensor
        src, src_lens, src_mask = self._pad(src)
        trg, trg_lens, trg_mask = self._pad(trg)
        # trg_target, _, _ = self._pad(trg_target)
        trg_oov, _, _ = self._pad(trg_oov)
        src_oov, _, _ = self._pad(src_oov)
        src_bow = self._pad_bow(src_bow)

        return src, src_lens, src_mask, trg, trg_lens, trg_mask, src_oov, trg_oov, oov_lists, src_bow

    def collate_fn_one2many(self, batches):
        assert self.type == 'one2many', 'The type of dataset should be one2many.'
        if self.remove_src_eos:
            # source with oov words replaced by <unk>
            src = [b['src'] for b in batches]
            # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
            src_oov = [b['src_oov'] for b in batches]
        else:
            # source with oov words replaced by <unk>
            src = [b['src'] + [self.word2idx[EOS_WORD]] for b in batches]
            # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
            src_oov = [b['src_oov'] + [self.word2idx[EOS_WORD]] for b in batches]

        batch_size = len(src)

        # trg: a list of concatenated targets, the targets in a concatenated target are separated by a delimiter, oov replaced by UNK
        # trg_oov: a list of concatenated targets, the targets in a concatenated target are separated by a delimiter, oovs are replaced with temporary idx, e.g. 50000, 50001 etc.)
        if self.load_train:
            trg = []
            trg_oov = []
            for b in batches:
                trg_concat = []
                trg_oov_concat = []
                trg_size = len(b['trg'])
                assert len(b['trg']) == len(b['trg_copy'])
                for trg_idx, (trg_phase, trg_phase_oov) in enumerate(zip(b['trg'], b[
                    'trg_copy'])):  # b['trg'] contains a list of targets, each target is a list of indices
                    # for trg_idx, a in enumerate(zip(b['trg'], b['trg_copy'])):
                    # trg_phase, trg_phase_oov = a
                    if trg_idx == trg_size - 1:  # if this is the last keyphrase, end with <eos>
                        trg_concat += trg_phase + [self.word2idx[EOS_WORD]]
                        trg_oov_concat += trg_phase_oov + [self.word2idx[EOS_WORD]]
                    else:
                        trg_concat += trg_phase + [
                            self.delimiter]  # trg_concat = [target_1] + [delimiter] + [target_2] + [delimiter] + ...
                        trg_oov_concat += trg_phase_oov + [self.delimiter]
                trg.append(trg_concat)
                trg_oov.append(trg_oov_concat)
        else:
            trg, trg_oov = None, None
        # trg = [[t + [self.word2idx[EOS_WORD]] for t in b['trg']] for b in batches]
        # trg_oov = [[t + [self.word2idx[EOS_WORD]] for t in b['trg_copy']] for b in batches]

        oov_lists = [b['oov_list'] for b in batches]
        src_bow = [b['src_bow'] for b in batches]

        # b['src_str'] is a word_list for source text, b['trg_str'] is a list of word list
        src_str = [b['src_str'] for b in batches]
        trg_str = [b['trg_str'] for b in batches]

        original_indices = list(range(batch_size))

        # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
        if self.load_train:
            seq_pairs = sorted(zip(src, src_oov, oov_lists, src_str, trg_str, trg, trg_oov, original_indices),
                               key=lambda p: len(p[0]), reverse=True)
            src, src_oov, oov_lists, src_str, trg_str, trg, trg_oov, original_indices = zip(*seq_pairs)
        else:
            seq_pairs = sorted(zip(src, src_oov, oov_lists, src_str, trg_str, original_indices, src_bow),
                               key=lambda p: len(p[0]), reverse=True)
            src, src_oov, oov_lists, src_str, trg_str, original_indices, src_bow = zip(*seq_pairs)

        # pad the src and target sequences with <pad> token and convert to LongTensor
        src, src_lens, src_mask = self._pad(src)
        src_oov, _, _ = self._pad(src_oov)
        if self.load_train:
            trg, trg_lens, trg_mask = self._pad(trg)
            trg_oov, _, _ = self._pad(trg_oov)
        else:
            trg_lens, trg_mask = None, None

        src_bow = self._pad_bow(src_bow)

        return src, src_lens, src_mask, src_oov, oov_lists, src_str, trg_str, trg, trg_oov, trg_lens, trg_mask, original_indices, src_bow


def build_dataset(src_trgs_pairs, word2idx, bow_dictionary, opt, mode='one2one', include_original=True):
    '''
    Standard process for copy model
    :param mode: one2one or one2many
    :param include_original: keep the original texts of source and target
    :return:
    '''
    return_examples = []
    oov_target = 0
    max_oov_len = 0
    zero_bow = 0

    for idx, (source, targets) in enumerate(src_trgs_pairs):
        src = [word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size
               else word2idx[UNK_WORD] for w in source]
        src_oov, oov_dict, oov_list = extend_vocab_OOV(source, word2idx, opt.vocab_size, opt.max_unk_words)

        examples = []  # for one-to-many

        for target in targets:
            example = {}

            if include_original:
                example['src_str'] = source
                example['trg_str'] = target

            example['src'] = src
            example['src_bow'] = bow_dictionary.doc2bow(source)
            if len(example['src_bow']) == 0:
                zero_bow += 1
                # for train and valid data, we do not account for zero bow, contrary for test
                if mode == "one2one":
                    continue
                # print("%d pairs have zero bow" % idx)

            trg = [word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size
                   else word2idx[UNK_WORD] for w in target]

            example['trg'] = trg

            example['src_oov'] = src_oov
            example['oov_dict'] = oov_dict
            example['oov_list'] = oov_list
            if len(oov_list) > max_oov_len:
                max_oov_len = len(oov_list)

            # oov words are replaced with new index
            trg_copy = []
            for w in target:
                if w in word2idx and word2idx[w] < opt.vocab_size:
                    trg_copy.append(word2idx[w])
                elif w in oov_dict:
                    trg_copy.append(oov_dict[w])
                else:
                    trg_copy.append(word2idx[UNK_WORD])

            example['trg_copy'] = trg_copy
            if any([w >= opt.vocab_size for w in trg_copy]):
                oov_target += 1

            if mode == 'one2one':
                return_examples.append(example)
            else:
                examples.append(example)

        if mode == 'one2many' and len(examples) > 0:
            o2m_example = {}
            keys = examples[0].keys()
            for key in keys:
                if key.startswith('src') or key.startswith('oov'):
                    o2m_example[key] = examples[0][key]
                else:
                    o2m_example[key] = [e[key] for e in examples]
            if include_original:
                assert len(o2m_example['src']) == len(o2m_example['src_oov']) == len(o2m_example['src_str'])
                assert len(o2m_example['oov_dict']) == len(o2m_example['oov_list'])
                assert len(o2m_example['trg']) == len(o2m_example['trg_copy']) == len(o2m_example['trg_str'])
            else:
                assert len(o2m_example['src']) == len(o2m_example['src_oov'])
                assert len(o2m_example['oov_dict']) == len(o2m_example['oov_list'])
                assert len(o2m_example['trg']) == len(o2m_example['trg_copy'])

            return_examples.append(o2m_example)

    print('Find #(oov_target)/#(all) = %d/%d' % (oov_target, len(return_examples)))
    print('Find max_oov_len = %d' % max_oov_len)
    print("Find %d zero bow" % zero_bow)

    return return_examples


def extend_vocab_OOV(source_words, word2idx, vocab_size, max_unk_words):
    """
    Map source words to their ids, including OOV words. Also return a list of OOVs in the article.
    WARNING: if the number of oovs in the source text is more than max_unk_words, ignore and replace them as <unk>
    Args:
        source_words: list of words (strings)
        word2idx: vocab word2idx
        vocab_size: the maximum acceptable index of word in vocab
    Returns:
        ids: A list of word ids (integers); OOVs are represented by their temporary article OOV number. If the vocabulary size is 50k and the article has 3 OOVs, then these temporary OOV numbers will be 50000, 50001, 50002.
        oovs: A list of the OOV words in the article (strings), in the order corresponding to their temporary article OOV numbers.
    """
    src_oov = []
    oov_dict = {}
    for w in source_words:
        if w in word2idx and word2idx[w] < vocab_size:  # a OOV can be either outside the vocab or id>=vocab_size
            src_oov.append(word2idx[w])
        else:
            if len(oov_dict) < max_unk_words:
                # e.g. 50000 for the first article OOV, 50001 for the second...
                word_id = oov_dict.get(w, len(oov_dict) + vocab_size)
                oov_dict[w] = word_id
                src_oov.append(word_id)
            else:
                # exceeds the maximum number of acceptable oov words, replace it with <unk>
                word_id = word2idx[UNK_WORD]
                src_oov.append(word_id)

    oov_list = [w for w, w_id in sorted(oov_dict.items(), key=lambda x: x[1])]
    return src_oov, oov_dict, oov_list
