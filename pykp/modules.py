import torch
import torch.nn as nn
from pykp.masked_softmax import MaskedSoftmax


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, bidirectional, pad_token, dropout=0.0):
        super(RNNEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pad_token = pad_token
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size,
            self.pad_token
        )
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=bidirectional, batch_first=True, dropout=dropout)

    def forward(self, src, src_lens):
        """
        :param src: [batch, src_seq_len]
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        :return:
        """
        src_embed = self.embedding(src)  # [batch, src_len, embed_size]
        packed_input_src = nn.utils.rnn.pack_padded_sequence(src_embed, src_lens, batch_first=True)
        memory_bank, encoder_final_state = self.rnn(packed_input_src)
        # ([batch, seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        memory_bank, _ = nn.utils.rnn.pad_packed_sequence(memory_bank, batch_first=True)  # unpack (back to padded)

        # only extract the final state in the last layer
        if self.bidirectional:
            encoder_last_layer_final_state = torch.cat((encoder_final_state[-1, :, :], encoder_final_state[-2, :, :]),
                                                       1)
            # [batch, hidden_size*2]
        else:
            encoder_last_layer_final_state = encoder_final_state[-1, :, :]  # [batch, hidden_size]

        return memory_bank.contiguous(), encoder_last_layer_final_state


class Attention(nn.Module):
    def __init__(self, decoder_size, memory_bank_size, coverage_attn, attn_mode):
        super(Attention, self).__init__()
        # attention
        if attn_mode == "concat":
            self.v = nn.Linear(decoder_size, 1, bias=False)
            self.decode_project = nn.Linear(decoder_size, decoder_size)
        self.memory_project = nn.Linear(memory_bank_size, decoder_size, bias=False)
        self.coverage_attn = coverage_attn
        if coverage_attn:
            self.coverage_project = nn.Linear(1, decoder_size, bias=False)
        self.softmax = MaskedSoftmax(dim=1)
        # self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.attn_mode = attn_mode

    def score(self, memory_bank, decoder_state, coverage=None):
        """
        :param memory_bank: [batch_size, max_input_seq_len, self.num_directions * self.encoder_size]
        :param decoder_state: [batch_size, decoder_size]
        :param coverage: [batch_size, max_input_seq_len]
        :return: score: [batch_size, max_input_seq_len]
        """
        batch_size, max_input_seq_len, memory_bank_size = list(memory_bank.size())
        decoder_size = decoder_state.size(1)

        if self.attn_mode == "general":
            # project memory_bank
            memory_bank_ = memory_bank.view(-1,
                                            memory_bank_size)  # [batch_size*max_input_seq_len, memory_bank_size]
            """
            if self.coverage_attn:
                coverage_input = coverage.view(-1, 1)  # [batch_size*max_input_seq_len, 1]
                memory_bank_ += self.coverage_project(coverage_input)  # [batch_size*max_input_seq_len, decoder_size]
                memory_bank_ = self.tanh(memory_bank_)

            encoder_feature = self.memory_project(memory_bank_)  # [batch_size*max_input_seq_len, decoder size]
            """

            encoder_feature = self.memory_project(memory_bank_)  # [batch_size*max_input_seq_len, decoder size]
            if self.coverage_attn:
                coverage_input = coverage.view(-1, 1)  # [batch_size*max_input_seq_len, 1]
                encoder_feature += self.coverage_project(coverage_input)  # [batch_size*max_input_seq_len, decoder_size]
                encoder_feature = self.tanh(encoder_feature)

            # expand decoder state
            decoder_state_expanded = decoder_state.unsqueeze(1).expand(batch_size, max_input_seq_len,
                                                                       decoder_size).contiguous()
            decoder_state_expanded = decoder_state_expanded.view(-1,
                                                                 decoder_size)  # [batch_size*max_input_seq_len, decoder_size]
            # Perform bi-linear operation
            scores = torch.bmm(decoder_state_expanded.unsqueeze(1),
                               encoder_feature.unsqueeze(2))  # [batch_size*max_input_seq_len, 1, 1]

        else:  # Bahdanau style attention
            # project memory_bank
            memory_bank_ = memory_bank.view(-1, memory_bank_size)  # [batch_size*max_input_seq_len, memory_bank_size]
            encoder_feature = self.memory_project(memory_bank_)  # [batch_size*max_input_seq_len, decoder size]
            # project decoder state
            dec_feature = self.decode_project(decoder_state)  # [batch_size, decoder_size]
            dec_feature_expanded = dec_feature.unsqueeze(1).expand(batch_size, max_input_seq_len,
                                                                   decoder_size).contiguous()
            dec_feature_expanded = dec_feature_expanded.view(-1,
                                                             decoder_size)  # [batch_size*max_input_seq_len, decoder_size]
            # sum up attention features
            att_features = encoder_feature + dec_feature_expanded  # [batch_size*max_input_seq_len, decoder_size]

            # Apply coverage
            if self.coverage_attn:
                coverage_input = coverage.view(-1, 1)  # [batch_size*max_input_seq_len, 1]
                coverage_feature = self.coverage_project(coverage_input)  # [batch_size*max_input_seq_len, decoder_size]
                # print(coverage.size())
                # print(coverage_feature.size())
                # print(att_features.size())
                att_features = att_features + coverage_feature

            # compute attention score and normalize them
            e = self.tanh(att_features)  # [batch_size*max_input_seq_len, decoder_size]
            scores = self.v(e)  # [batch_size*max_input_seq_len, 1]

        scores = scores.view(-1, max_input_seq_len)  # [batch_size, max_input_seq_len]
        return scores

    def forward(self, decoder_state, memory_bank, src_mask=None, coverage=None):
        """
        :param decoder_state: [batch_size, decoder_size]
        :param memory_bank: [batch_size, max_input_seq_len, self.num_directions * self.encoder_size]
        :param src_mask: [batch_size, max_input_seq_len]
        :param coverage: [batch_size, max_input_seq_len]
        :return: context: [batch_size, self.num_directions * self.encoder_size], attn_dist: [batch_size, max_input_seq_len], coverage: [batch_size, max_input_seq_len]
        """
        # init dimension info
        batch_size, max_input_seq_len, memory_bank_size = list(memory_bank.size())
        # decoder_size = decoder_state.size(1)

        if src_mask is None:  # if it does not supply a source mask, create a dummy mask with all ones
            src_mask = memory_bank.new_ones(batch_size, max_input_seq_len)

        """
        # project memory_bank
        memory_bank = memory_bank.view(-1, memory_bank_size)  # [batch_size*max_input_seq_len, memory_bank_size]
        encoder_feature = self.memory_project(memory_bank)  # [batch_size*max_input_seq_len, decoder size]

        # project decoder state
        dec_feature = self.decode_project(decoder_state)  # [batch_size, decoder_size]
        dec_feature_expanded = dec_feature.unsqueeze(1).expand(batch_size, max_input_seq_len, decoder_size).contiguous()
        dec_feature_expanded = dec_feature_expanded.view(-1, decoder_size)  # [batch_size*max_input_seq_len, decoder_size]

        # sum up attention features
        att_features = encoder_feature + dec_feature_expanded  # [batch_size*max_input_seq_len, decoder_size]

        # Apply coverage
        if self.coverage_attn:
            coverage_input = coverage.view(-1, 1)  # [batch_size*max_input_seq_len, 1]
            coverage_feature = self.coverage_project(coverage_input)  # [batch_size*max_input_seq_len, decoder_size]
            #print(coverage.size())
            #print(coverage_feature.size())
            #print(att_features.size())
            att_features = att_features + coverage_feature

        # compute attention score and normalize them
        e = self.tanh(att_features)  # [batch_size*max_input_seq_len, decoder_size]
        scores = self.v(e)  # [batch_size*max_input_seq_len, 1]
        scores = scores.view(-1, max_input_seq_len)  # [batch_size, max_input_seq_len]
        """

        scores = self.score(memory_bank, decoder_state, coverage)
        attn_dist = self.softmax(scores, mask=src_mask)

        # Compute weighted sum of memory bank features
        attn_dist = attn_dist.unsqueeze(1)  # [batch_size, 1, max_input_seq_len]
        memory_bank = memory_bank.view(-1, max_input_seq_len,
                                       memory_bank_size)  # batch_size, max_input_seq_len, memory_bank_size]
        context = torch.bmm(attn_dist, memory_bank)  # [batch_size, 1, memory_bank_size]
        context = context.squeeze(1)  # [batch_size, memory_bank_size]
        attn_dist = attn_dist.squeeze(1)  # [batch_size, max_input_seq_len]

        # Update coverage
        if self.coverage_attn:
            coverage = coverage.view(-1, max_input_seq_len)
            coverage = coverage + attn_dist
            assert coverage.size() == torch.Size([batch_size, max_input_seq_len])

        assert attn_dist.size() == torch.Size([batch_size, max_input_seq_len])
        assert context.size() == torch.Size([batch_size, memory_bank_size])

        return context, attn_dist, coverage


class TopicAttention(nn.Module):
    def __init__(self, decoder_size, memory_bank_size, coverage_attn, attn_mode, topic_num):
        super(TopicAttention, self).__init__()
        # attention
        if attn_mode == "concat":
            self.v = nn.Linear(decoder_size, 1, bias=False)
            self.decode_project = nn.Linear(decoder_size, decoder_size)
        self.topic_num = topic_num
        self.memory_project = nn.Linear(memory_bank_size, decoder_size, bias=False)
        self.topic_project = nn.Linear(topic_num, decoder_size, bias=False)
        self.coverage_attn = coverage_attn
        if coverage_attn:
            self.coverage_project = nn.Linear(1, decoder_size, bias=False)
        self.softmax = MaskedSoftmax(dim=1)
        # self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.attn_mode = attn_mode

    def score(self, memory_bank, decoder_state, topic_represent, coverage=None):
        """
        :param memory_bank: [batch_size, max_input_seq_len, self.num_directions * self.encoder_size]
        :param decoder_state: [batch_size, decoder_size]
        :param topic_represent: [batch_size, topic_num]
        :param coverage: [batch_size, max_input_seq_len]
        :return: score: [batch_size, max_input_seq_len]
        """
        batch_size, max_input_seq_len, memory_bank_size = list(memory_bank.size())
        decoder_size = decoder_state.size(1)

        if self.attn_mode == "general":
            # project memory_bank
            memory_bank_ = memory_bank.view(-1,
                                            memory_bank_size)  # [batch_size*max_input_seq_len, memory_bank_size]

            encoder_feature = self.memory_project(memory_bank_)  # [batch_size*max_input_seq_len, decoder size]
            if self.coverage_attn:
                coverage_input = coverage.view(-1, 1)  # [batch_size*max_input_seq_len, 1]
                encoder_feature += self.coverage_project(coverage_input)  # [batch_size*max_input_seq_len, decoder_size]
                encoder_feature = self.tanh(encoder_feature)

            # expand decoder state
            decoder_state_expanded = decoder_state.unsqueeze(1).expand(batch_size, max_input_seq_len,
                                                                       decoder_size).contiguous()
            decoder_state_expanded = decoder_state_expanded.view(-1,
                                                                 decoder_size)  # [batch_size*max_input_seq_len, decoder_size]
            # Perform bi-linear operation
            scores = torch.bmm(decoder_state_expanded.unsqueeze(1),
                               encoder_feature.unsqueeze(2))  # [batch_size*max_input_seq_len, 1, 1]

        else:  # Bahdanau style attention
            # project memory_bank
            memory_bank_ = memory_bank.view(-1, memory_bank_size)  # [batch_size*max_input_seq_len, memory_bank_size]
            encoder_feature = self.memory_project(memory_bank_)  # [batch_size*max_input_seq_len, decoder size]
            # project decoder state
            topic_feature = self.topic_project(topic_represent)  # [batch_size, decoder_size]
            topic_feature_expanded = topic_feature.unsqueeze(1).expand(batch_size, max_input_seq_len,
                                                                       decoder_size).contiguous()
            topic_feature_expanded = topic_feature_expanded.view(-1,
                                                                 decoder_size)  # [batch_size*max_input_seq_len, decoder_size]

            dec_feature = self.decode_project(decoder_state)  # [batch_size, decoder_size]
            dec_feature_expanded = dec_feature.unsqueeze(1).expand(batch_size, max_input_seq_len,
                                                                   decoder_size).contiguous()
            dec_feature_expanded = dec_feature_expanded.view(-1,
                                                             decoder_size)  # [batch_size*max_input_seq_len, decoder_size]
            # sum up attention features
            att_features = encoder_feature + dec_feature_expanded + topic_feature_expanded  # [batch_size*max_input_seq_len, decoder_size]

            # Apply coverage
            if self.coverage_attn:
                coverage_input = coverage.view(-1, 1)  # [batch_size*max_input_seq_len, 1]
                coverage_feature = self.coverage_project(coverage_input)  # [batch_size*max_input_seq_len, decoder_size]
                # print(coverage.size())
                # print(coverage_feature.size())
                # print(att_features.size())
                att_features = att_features + coverage_feature

            # compute attention score and normalize them
            e = self.tanh(att_features)  # [batch_size*max_input_seq_len, decoder_size]
            scores = self.v(e)  # [batch_size*max_input_seq_len, 1]

        scores = scores.view(-1, max_input_seq_len)  # [batch_size, max_input_seq_len]
        return scores

    def forward(self, decoder_state, memory_bank, topic_represent, src_mask=None, coverage=None):
        """
        :param decoder_state: [batch_size, decoder_size]
        :param memory_bank: [batch_size, max_input_seq_len, self.num_directions * self.encoder_size]
        :param src_mask: [batch_size, max_input_seq_len]
        :param coverage: [batch_size, max_input_seq_len]
        :return: context: [batch_size, self.num_directions * self.encoder_size], attn_dist: [batch_size, max_input_seq_len], coverage: [batch_size, max_input_seq_len]
        """
        # init dimension info
        batch_size, max_input_seq_len, memory_bank_size = list(memory_bank.size())

        if src_mask is None:  # if it does not supply a source mask, create a dummy mask with all ones
            src_mask = memory_bank.new_ones(batch_size, max_input_seq_len)

        scores = self.score(memory_bank, decoder_state, topic_represent, coverage)
        attn_dist = self.softmax(scores, mask=src_mask)

        # Compute weighted sum of memory bank features
        attn_dist = attn_dist.unsqueeze(1)  # [batch_size, 1, max_input_seq_len]
        memory_bank = memory_bank.view(-1, max_input_seq_len,
                                       memory_bank_size)  # batch_size, max_input_seq_len, memory_bank_size]
        context = torch.bmm(attn_dist, memory_bank)  # [batch_size, 1, memory_bank_size]
        context = context.squeeze(1)  # [batch_size, memory_bank_size]
        attn_dist = attn_dist.squeeze(1)  # [batch_size, max_input_seq_len]

        # Update coverage
        if self.coverage_attn:
            coverage = coverage.view(-1, max_input_seq_len)
            coverage = coverage + attn_dist
            assert coverage.size() == torch.Size([batch_size, max_input_seq_len])

        assert attn_dist.size() == torch.Size([batch_size, max_input_seq_len])
        assert context.size() == torch.Size([batch_size, memory_bank_size])

        return context, attn_dist, coverage


class RNNDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, memory_bank_size, coverage_attn, copy_attn,
                 review_attn, pad_idx, attn_mode, dropout=0.0, use_topic_represent=False, topic_attn=False,
                 topic_attn_in=False, topic_copy=False, topic_dec=False, topic_num=50):
        super(RNNDecoder, self).__init__()
        self.use_topic_represent = use_topic_represent
        self.topic_attn = topic_attn
        self.topic_attn_in = topic_attn_in
        self.topic_copy = topic_copy
        self.topic_dec = topic_dec
        self.topic_num = topic_num

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.memory_bank_size = memory_bank_size
        self.dropout = nn.Dropout(dropout)
        self.coverage_attn = coverage_attn
        self.copy_attn = copy_attn
        self.review_attn = review_attn
        self.pad_token = pad_idx
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size,
            self.pad_token
        )

        self.input_size = embed_size

        if use_topic_represent:
            if topic_dec:
                self.input_size = embed_size + topic_num

        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=False, batch_first=False, dropout=dropout)
        if topic_attn_in:
            self.attention_layer = TopicAttention(
                decoder_size=hidden_size,
                memory_bank_size=memory_bank_size,
                coverage_attn=coverage_attn,
                attn_mode=attn_mode,
                topic_num=topic_num
            )
        else:
            self.attention_layer = Attention(
                decoder_size=hidden_size,
                memory_bank_size=memory_bank_size,
                coverage_attn=coverage_attn,
                attn_mode=attn_mode
            )
        if copy_attn:
            if topic_copy:
                self.p_gen_linear = nn.Linear(embed_size + hidden_size + memory_bank_size + topic_num, 1)
            else:
                self.p_gen_linear = nn.Linear(embed_size + hidden_size + memory_bank_size, 1)

        self.sigmoid = nn.Sigmoid()

        if topic_attn:
            self.vocab_dist_linear_1 = nn.Linear(hidden_size + memory_bank_size + topic_num, hidden_size)
        else:
            self.vocab_dist_linear_1 = nn.Linear(hidden_size + memory_bank_size, hidden_size)

        self.vocab_dist_linear_2 = nn.Linear(hidden_size, vocab_size)
        self.softmax = MaskedSoftmax(dim=1)

    def forward(self, y, topic_represent, h, memory_bank, src_mask, max_num_oovs, src_oov, coverage):
        """
        :param y: [batch_size]
        :param h: [num_layers, batch_size, decoder_size]
        :param memory_bank: [batch_size, max_src_seq_len, memory_bank_size]
        :param src_mask: [batch_size, max_src_seq_len]
        :param max_num_oovs: int
        :param src_oov: [batch_size, max_src_seq_len]
        :param coverage: [batch_size, max_src_seq_len]
        :return:
        """
        batch_size, max_src_seq_len = list(src_oov.size())
        assert y.size() == torch.Size([batch_size])
        if self.use_topic_represent:
            assert topic_represent.size() == torch.Size([batch_size, self.topic_num])
        assert h.size() == torch.Size([self.num_layers, batch_size, self.hidden_size])

        # init input embedding
        y_emb = self.embedding(y).unsqueeze(0)  # [1, batch_size, embed_size]

        if self.use_topic_represent and self.topic_dec:
            rnn_input = torch.cat([y_emb, topic_represent.unsqueeze(0)], dim=2)
        else:
            rnn_input = y_emb

        _, h_next = self.rnn(rnn_input, h)

        assert h_next.size() == torch.Size([self.num_layers, batch_size, self.hidden_size])

        last_layer_h_next = h_next[-1, :, :]  # [batch, decoder_size]

        # apply attention, get input-aware context vector, attention distribution and update the coverage vector
        if self.topic_attn_in:
            context, attn_dist, coverage = self.attention_layer(last_layer_h_next, memory_bank, topic_represent,
                                                                src_mask, coverage)
        else:
            context, attn_dist, coverage = self.attention_layer(last_layer_h_next, memory_bank, src_mask, coverage)
        # context: [batch_size, memory_bank_size]
        # attn_dist: [batch_size, max_input_seq_len]
        # coverage: [batch_size, max_input_seq_len]
        assert context.size() == torch.Size([batch_size, self.memory_bank_size])
        assert attn_dist.size() == torch.Size([batch_size, max_src_seq_len])

        if self.coverage_attn:
            assert coverage.size() == torch.Size([batch_size, max_src_seq_len])

        if self.topic_attn:
            vocab_dist_input = torch.cat((context, last_layer_h_next, topic_represent), dim=1)
            # [B, memory_bank_size + decoder_size + topic_num]
        else:
            vocab_dist_input = torch.cat((context, last_layer_h_next), dim=1)
            # [B, memory_bank_size + decoder_size]

        vocab_dist = self.softmax(self.vocab_dist_linear_2(self.dropout(self.vocab_dist_linear_1(vocab_dist_input))))

        p_gen = None
        if self.copy_attn:
            if self.topic_copy:
                p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0), topic_represent),
                                        dim=1)  # [B, memory_bank_size + decoder_size + embed_size]
            else:
                p_gen_input = torch.cat((context, last_layer_h_next, y_emb.squeeze(0)),
                                        dim=1)  # [B, memory_bank_size + decoder_size + embed_size]

            p_gen = self.sigmoid(self.p_gen_linear(p_gen_input))

            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if max_num_oovs > 0:
                extra_zeros = vocab_dist_.new_zeros((batch_size, max_num_oovs))
                vocab_dist_ = torch.cat((vocab_dist_, extra_zeros), dim=1)

            final_dist = vocab_dist_.scatter_add(1, src_oov, attn_dist_)
            assert final_dist.size() == torch.Size([batch_size, self.vocab_size + max_num_oovs])
        else:
            final_dist = vocab_dist
            assert final_dist.size() == torch.Size([batch_size, self.vocab_size])

        return final_dist, h_next, context, attn_dist, p_gen, coverage
