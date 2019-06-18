import torch
import penalties
import logging

class Beam:
    def __init__(self, size, pad, bos, eos,
                 n_best=1, cuda=False,
                 global_scorer=None,
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 exclusion_tokens=set(), max_eos_per_output_seq=1):
        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size)
                            .fill_(pad)]
        self.next_ys[0][0] = bos

        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}

        # Minimum prediction length
        self.min_length = min_length

        # Apply Penalty at every step
        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens

        self.eos_counters = torch.zeros(size, dtype=torch.long).to(self.next_ys[0].device)  # Store the number of emitted eos token for each hypothesis sequence
        self.max_eos_per_output_seq = max_eos_per_output_seq  # The max. number of eos token that a hypothesis sequence can have

    def get_current_tokens(self):
        """Get the outputs for the current timestep."""
        return self.next_ys[-1]

    def get_current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prev_ks[-1]

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def get_hyp(self, timestep, k):
        """
        walk back to construct the full hypothesis given the finished time step and beam idx
        :param timestep: int
        :param k: int
        :return:
        """
        hyp, attn = [], []
        # iterate from output sequence length (with eos but not bos) - 1 to 0f
        for j in range(len(self.prev_ks[:timestep]) -1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])  # j+1 so that it will iterate from the <eos> token, and end before the <bos>
            attn.append(self.attn[j][k])  # since it does not has attn for bos, it will also iterate from the attn for <eos>
            # attn[j][k] Tensor with size [src_len]
            k = self.prev_ks[j][k]  # find the beam idx of the previous token

        # hyp[::-1]: a list of idx (zero dim tensor), with len = output sequence length
        # torch.stack(attn): FloatTensor, with size: [output sequence length, src_len]
        return hyp[::-1], torch.stack(attn)

    def advance(self, word_logits, attn_dist):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.

        Parameters:

        * `word_logit`- probs of advancing from the last step [beam_size, vocab_size]
        * `attn_dist`- attention at the last step [beam_size, src_len]

        Returns: True if beam search is complete.
        """
        vocab_size = word_logits.size(1)
        # To be implemented: stepwise penalty

        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_logits)):
                word_logits[k][self._eos] = -1e20
        # Sum the previous scores
        if len(self.prev_ks) > 0:
            beam_scores = word_logits + self.scores.unsqueeze(1).expand_as(word_logits)
            # Don't let EOS have children. If it have reached the max number of eos.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos and self.eos_counters[i] >= self.max_eos_per_output_seq:
                    beam_scores[i] = -1e20
            # To be implemented: block n-gram repeated
            if self.block_ngram_repeat > 0:
                ngrams = []
                le = len(self.next_ys)
                for j in range(self.next_ys[-1].size(0)):
                    hyp, _ = self.get_hyp(le - 1, j)
                    ngrams = set()
                    fail = False
                    gram = []
                    for i in range(le - 1):
                        # Last n tokens, n = block_ngram_repeat
                        gram = (gram +
                                [hyp[i].item()])[-self.block_ngram_repeat:]
                        # Skip the blocking if it is in the exclusion list
                        if set(gram) & self.exclusion_tokens:
                            continue
                        if tuple(gram) in ngrams:
                            fail = True
                        ngrams.add(tuple(gram))
                    if fail:
                        beam_scores[j] = -10e20

        else:  # This is the first decoding step, every beam are the same
            beam_scores = word_logits[0]
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_idx = flat_beam_scores.topk(self.size, 0, True, True)  # [beam_size]

        self.all_scores.append(self.scores)  # list of tensor with size [beam_size]
        self.scores = best_scores

        # best_scores_idx indicate the idx in the flattened beam * vocab_size array, so need to convert
        # the idx back to which beam and word each score came from.
        prev_k = best_scores_idx / vocab_size  # convert it to the beam indices that the top k scores came from, LongTensor, size: [beam_size]
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_idx - prev_k * vocab_size))  # convert it to the vocab indices, LongTensor, size: [beam_size]
        self.attn.append(attn_dist.index_select(0, prev_k))  # select the attention dist from the corresponding beam, size: [beam_size, src_len]
        self.global_scorer.update_global_state(self)  # update coverage vector, previous coverage penalty, and cov_total
        self.update_eos_counter()  # update the eos_counter according to prev_ks

        for i in range(self.next_ys[-1].size(0)):  # For each generated token in the current step, check if it is EOS
            if self.next_ys[-1][i] == self._eos:
                self.eos_counters[i] += 1
                if self.eos_counters[i] == self.max_eos_per_output_seq:  # compute the score penalize by length and coverage amd append add it to finished
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
                    self.finished.append((s, len(self.next_ys) - 1, i))  # penalized score, length of sequence, beam_idx
            """
            elif self.next_ys[-1][i] == self._unk:  # if it is unk, replace it with the w
                _, max_attn_score_idx = self.attn[-1][i].max(0)
                self.next_ys[-1][i] = max_attn_score_idx
            """
        # End condition is when top-of-beam is EOS (and its number of EOS tokens reached the max) and no global score.
        if self.next_ys[-1][0] == self._eos and self.eos_counters[0] == self.max_eos_per_output_seq:
            self.all_scores.append(self.scores)
            self.eos_top = True

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs in the finished list
            while len(self.finished) < minimum:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys)-1, i)) # score, length of sequence (include eos but not bos), beam_idx
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t,k) for _, t, k in self.finished]
        return scores, ks

    def update_eos_counter(self):
        # update the eos_counter according to prev_ks
        self.eos_counters = self.eos_counters.index_select(0, self.prev_ks[-1])


class GNMTGlobalScorer:
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`

    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """

    def __init__(self, alpha, beta, cov_penalty, length_penalty):
        self.alpha = alpha
        self.beta = beta
        penalty_builder = penalties.PenaltyBuilder(cov_penalty, length_penalty)
        # Term will be subtracted from probability
        self.cov_penalty = penalty_builder.coverage_penalty()
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty()

    def score(self, beam, logprobs):
        """
        Rescores all the prediction scores of a beam based on penalty functions
        Return: normalized_probs, size: [beam_size]
        """
        normalized_probs = self.length_penalty(beam,
                                               logprobs,
                                               self.alpha)
        if not beam.stepwise_penalty:
            penalty = self.cov_penalty(beam,
                                       beam.global_state["coverage"],
                                       self.beta)
            normalized_probs -= penalty

        return normalized_probs

    def update_score(self, beam, attn):
        """
        Function to update scores of a Beam that is not finished
        """
        if "prev_penalty" in beam.global_state.keys():
            beam.scores.add_(beam.global_state["prev_penalty"])
            penalty = self.cov_penalty(beam,
                                       beam.global_state["coverage"] + attn,
                                       self.beta)
            beam.scores.sub_(penalty)

    def update_global_state(self, beam):
        """
        Keeps the coverage vector as sum of attentions
        """
        if len(beam.prev_ks) == 1:
            beam.global_state["prev_penalty"] = beam.scores.clone().fill_(0.0)  # [beam_size]
            beam.global_state["coverage"] = beam.attn[-1]  # [beam_size, src_len]
            self.cov_total = beam.attn[-1].sum(1)  # [beam_size], accumulate the penalty term for coverage
        else:
            self.cov_total += torch.min(beam.attn[-1],
                                        beam.global_state['coverage']).sum(1)
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])  # accumulate coverage vector

            prev_penalty = self.cov_penalty(beam,
                                            beam.global_state["coverage"],
                                            self.beta)
            beam.global_state["prev_penalty"] = prev_penalty

