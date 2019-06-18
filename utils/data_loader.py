import torch
import logging
from pykp.io import KeyphraseDataset
from torch.utils.data import DataLoader


def load_vocab(opt):
    # load vocab
    logging.info("Loading vocab from disk: %s" % (opt.vocab))
    if not opt.custom_vocab_filename_suffix:
        word2idx, idx2word, vocab, bow_dictionary = torch.load(opt.vocab + '/vocab.pt', 'wb')
    else:
        word2idx, idx2word, vocab = torch.load(opt.vocab + '/vocab.%s.pt' % opt.vocab_filename_suffix, 'wb')
    # assign vocab to opt
    opt.word2idx = word2idx
    opt.idx2word = idx2word
    opt.vocab = vocab
    opt.bow_dictionary = bow_dictionary
    logging.info('#(vocab)=%d' % len(vocab))
    logging.info('#(vocab used)=%d' % opt.vocab_size)
    logging.info('#(bow dictionary size)=%d' % len(bow_dictionary))

    return word2idx, idx2word, vocab, bow_dictionary


def load_data_and_vocab(opt, load_train=True):
    # load vocab
    word2idx, idx2word, vocab, bow_dictionary = load_vocab(opt)

    # constructor data loader
    logging.info("Loading train and validate data from '%s'" % opt.data)

    if load_train:  # load training dataset
        if not opt.one2many:  # load one2one dataset
            if not opt.custom_data_filename_suffix:
                train_one2one = torch.load(opt.data + '/train.one2one.pt', 'wb')
            else:
                train_one2one = torch.load(opt.data + '/train.one2one.%s.pt' % opt.data_filename_suffix, 'wb')
            train_one2one_dataset = KeyphraseDataset(train_one2one, word2idx=word2idx, idx2word=idx2word, bow_dictionary=bow_dictionary,
                                                     type='one2one', load_train=load_train, remove_src_eos=opt.remove_src_eos)
            train_loader = DataLoader(dataset=train_one2one_dataset,
                                      collate_fn=train_one2one_dataset.collate_fn_one2one,
                                      num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                                      shuffle=True)
            train_bow_loader = DataLoader(dataset=train_one2one_dataset,
                                      collate_fn=train_one2one_dataset.collate_bow,
                                      num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                                      shuffle=True)
            logging.info('#(train data size: #(batch)=%d' % (len(train_loader)))

            if not opt.custom_data_filename_suffix:
                valid_one2one = torch.load(opt.data + '/valid.one2one.pt', 'wb')
            else:
                valid_one2one = torch.load(opt.data + '/valid.one2one.%s.pt' % opt.data_filename_suffix, 'wb')
            valid_one2one_dataset = KeyphraseDataset(valid_one2one, word2idx=word2idx, idx2word=idx2word, bow_dictionary=bow_dictionary,
                                                     type='one2one', load_train=load_train, remove_src_eos=opt.remove_src_eos)
            valid_loader = DataLoader(dataset=valid_one2one_dataset,
                                      collate_fn=valid_one2one_dataset.collate_fn_one2one,
                                      num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                                      shuffle=False)
            valid_bow_loader = DataLoader(dataset=valid_one2one_dataset,
                                          collate_fn=valid_one2one_dataset.collate_bow,
                                          num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                                          shuffle=False)
            logging.info('#(valid data size: #(batch)=%d' % (len(valid_loader)))

        else:  # load one2many dataset
            if not opt.custom_data_filename_suffix:
                train_one2many = torch.load(opt.data + '/train.one2many.pt', 'wb')
            else:
                train_one2many = torch.load(opt.data + '/train.one2many.%s.pt' % opt.data_filename_suffix, 'wb')
            train_one2many_dataset = KeyphraseDataset(train_one2many, word2idx=word2idx, idx2word=idx2word, type='one2many', delimiter_type=opt.delimiter_type, load_train=load_train, remove_src_eos=opt.remove_src_eos)
            train_loader = DataLoader(dataset=train_one2many_dataset,
                                      collate_fn=train_one2many_dataset.collate_fn_one2many,
                                      num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                                      shuffle=True)
            logging.info('#(train data size: #(batch)=%d' % (len(train_loader)))

            if not opt.custom_data_filename_suffix:
                valid_one2many = torch.load(opt.data + '/valid.one2many.pt', 'wb')
            else:
                valid_one2many = torch.load(opt.data + '/valid.one2many.%s.pt' % opt.data_filename_suffix, 'wb')
            #valid_one2many = valid_one2many[:2000]
            valid_one2many_dataset = KeyphraseDataset(valid_one2many, word2idx=word2idx, idx2word=idx2word,
                                                      type='one2many', delimiter_type=opt.delimiter_type, load_train=load_train, remove_src_eos=opt.remove_src_eos)
            valid_loader = DataLoader(dataset=valid_one2many_dataset,
                                      collate_fn=valid_one2many_dataset.collate_fn_one2many,
                                      num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                                      shuffle=False)
            logging.info('#(valid data size: #(batch)=%d' % (len(valid_loader)))
        return train_loader, train_bow_loader, valid_loader, valid_bow_loader, word2idx, idx2word, vocab, bow_dictionary
    else:
        if not opt.custom_data_filename_suffix:
            test_one2many = torch.load(opt.data + '/test.one2many.pt', 'wb')
        else:
            test_one2many = torch.load(opt.data + '/test.one2many.%s.pt' % opt.data_filename_suffix, 'wb')
        test_one2many_dataset = KeyphraseDataset(test_one2many, word2idx=word2idx, idx2word=idx2word, bow_dictionary=bow_dictionary,
                                                      type='one2many', delimiter_type=opt.delimiter_type, load_train=load_train, remove_src_eos=opt.remove_src_eos)
        test_loader = DataLoader(dataset=test_one2many_dataset,
                                      collate_fn=test_one2many_dataset.collate_fn_one2many,
                                      num_workers=opt.batch_workers, batch_size=opt.batch_size, pin_memory=True,
                                      shuffle=False)
        logging.info('#(test data size: #(batch)=%d' % (len(test_loader)))

        return test_loader, word2idx, idx2word, vocab, bow_dictionary
