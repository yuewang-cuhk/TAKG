import math
import time


class LossStatistics:
    """
    Accumulator for loss staistics. Modified from OpenNMT
    """

    def __init__(self, loss=0.0, n_tokens=0, n_batch=0, forward_time=0.0, loss_compute_time=0.0, backward_time=0.0):
        assert type(loss) is float or type(loss) is int
        assert type(n_tokens) is int
        self.loss = loss
        if math.isnan(loss):
            raise ValueError("Loss is NaN")
        self.n_tokens = n_tokens
        self.n_batch = n_batch
        self.forward_time = forward_time
        self.loss_compute_time = loss_compute_time
        self.backward_time = backward_time

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `LossStatistics` object

        Args:
            stat: another statistic object
        """
        self.loss += stat.loss
        if math.isnan(stat.loss):
            raise ValueError("Loss is NaN")
        self.n_tokens += stat.n_tokens
        self.n_batch += stat.n_batch
        self.forward_time += stat.forward_time
        self.loss_compute_time += stat.loss_compute_time
        self.backward_time += stat.backward_time

    def xent(self):
        """ compute normalized cross entropy """
        assert self.n_tokens > 0, "n_tokens must be larger than 0"
        return self.loss / self.n_tokens

    def ppl(self):
        """ compute normalized perplexity """
        assert self.n_tokens > 0, "n_tokens must be larger than 0"
        return math.exp(min(self.loss / self.n_tokens, 100))

    def total_time(self):
        return self.forward_time, self.loss_compute_time, self.backward_time

    def clear(self):
        self.loss = 0.0
        self.n_tokens = 0
        self.n_batch = 0
        self.forward_time = 0.0
        self.loss_compute_time = 0.0
        self.backward_time = 0.0


class RewardStatistics:
    """
    Accumulator for reward staistics.
    """
    def __init__(self, final_reward=0.0, pg_loss=0.0, n_batch=0, sample_time=0, q_estimate_compute_time=0, backward_time=0):
        self.final_reward = final_reward
        self.pg_loss = pg_loss
        if math.isnan(pg_loss):
            raise ValueError("Policy gradient loss is NaN")
        self.n_batch = n_batch
        self.sample_time = sample_time
        self.q_estimate_compute_time = q_estimate_compute_time
        self.backward_time = backward_time

    def update(self, stat):
        self.final_reward += stat.final_reward
        if math.isnan(stat.pg_loss):
            raise ValueError("Policy gradient loss is NaN")
        self.pg_loss += stat.pg_loss
        self.n_batch += stat.n_batch
        self.sample_time += stat.sample_time
        self.q_estimate_compute_time += stat.q_estimate_compute_time
        self.backward_time += stat.backward_time

    def reward(self):
        assert self.n_batch > 0, "n_batch must be positive"
        return self.final_reward / self.n_batch

    def loss(self):
        assert self.n_batch > 0, "n_batch must be positive"
        return self.pg_loss / self.n_batch

    def total_time(self):
        return self.sample_time, self.q_estimate_compute_time, self.backward_time

    def clear(self):
        self.final_reward = 0.0
        self.pg_loss = 0.0
        self.n_batch = 0.0
        self.sample_time = 0.0
        self.q_estimate_compute_time = 0.0
        self.backward_time = 0.0
