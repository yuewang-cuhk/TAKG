from nltk.stem.porter import *
stemmer = PorterStemmer()
import pykp

def prediction_to_sentence(prediction, idx2word, vocab_size, oov, eos_idx, unk_idx=None, replace_unk=False, src_word_list=None, attn_dist=None):
    """
    :param prediction: a list of 0 dim tensor
    :param attn_dist: tensor with size [trg_len, src_len]
    :return: a list of words, does not include the final EOS
    """
    sentence = []
    for i, pred in enumerate(prediction):
        _pred = int(pred.item())  # convert zero dim tensor to int
        if i == len(prediction) - 1 and _pred == eos_idx:  # ignore the final EOS token
            break
        if _pred < vocab_size:
            if _pred == unk_idx and replace_unk:
                assert src_word_list is not None and attn_dist is not None, "If you need to replace unk, you must supply src_word_list and attn_dist"
                #_, max_attn_idx = attn_dist[i].max(0)
                _, max_attn_idx = attn_dist[i].topk(2, dim=0)
                if max_attn_idx[0] < len(src_word_list):
                    word = src_word_list[int(max_attn_idx[0].item())]
                else:
                    word = src_word_list[int(max_attn_idx[1].item())]
                    #word = pykp.io.EOS_WORD
            else:
                word = idx2word[_pred]
        else:
            word = oov[_pred - vocab_size]
        sentence.append(word)

    return sentence

def stem_str_list(str_list):
    # stem every word in a list of word list
    # str_list is a list of word list
    stemmed_str_list = []
    for word_list in str_list:
        stemmed_word_list = stem_word_list(word_list)
        stemmed_str_list.append(stemmed_word_list)
    return stemmed_str_list

def stem_word_list(word_list):
    return [stemmer.stem(w.strip().lower()) for w in word_list]

def split_concated_keyphrases(word_list, delimiter_word):
    """
    :param word_list: word list of concated keyprhases, separated by a delimiter
    :param delimiter_word
    :return: a list of keyphrases from a concated sequence, each keyphrase is a word list
    """
    tmp_pred_str_list = []
    tmp_word_list = []
    for word in word_list:
        if word != delimiter_word:
            tmp_word_list.append(word)
        else:
            if len(tmp_word_list) > 0:
                tmp_pred_str_list.append(tmp_word_list)
                tmp_word_list = []
    if len(tmp_word_list) > 0:  # append the final keyphrase to the pred_str_list
        tmp_pred_str_list.append(tmp_word_list)
    return tmp_pred_str_list

