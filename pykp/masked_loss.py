import torch
import math
import logging

EPS = 1e-8


def masked_cross_entropy(class_dist, target, trg_mask, trg_lens=None,
                         coverage_attn=False, coverage=None, attn_dist=None, lambda_coverage=0, coverage_loss=False,
                         delimiter_hidden_states=None, orthogonal_loss=False, lambda_orthogonal=0,
                         delimiter_hidden_states_lens=None):
    """
    :param class_dist: [batch_size, trg_seq_len, num_classes]
    :param target: [batch_size, trg_seq_len]
    :param trg_mask: [batch_size, trg_seq_len]
    :param trg_lens: a list with len of batch_size
    :param coverage_attn: boolean, whether to include coverage loss
    :param coverage: [batch_size, trg_seq_len, src_seq_len]
    :param attn_dist: [batch_size, trg_seq_len, src_seq_len]
    :param lambda_coverage: scalar, coefficient for coverage loss
    :param delimiter_hidden_states: [batch_size, decoder_size, num_delimiter]
    :return:
    """
    num_classes = class_dist.size(2)
    class_dist_flat = class_dist.view(-1, num_classes)  # [batch_size*trg_seq_len, num_classes]
    log_dist_flat = torch.log(class_dist_flat + EPS)
    target_flat = target.view(-1, 1)  # [batch*trg_seq_len, 1]
    losses_flat = -torch.gather(log_dist_flat, dim=1, index=target_flat)  # [batch * trg_seq_len, 1]
    losses = losses_flat.view(*target.size())  # [batch, trg_seq_len]
    if coverage_attn and coverage_loss:
        coverage_losses = compute_coverage_losses(coverage, attn_dist)
        losses = losses + lambda_coverage * coverage_losses
    if trg_mask is not None:
        losses = losses * trg_mask
    '''
    if divided_by_seq_len:
        trg_lens_tensor = torch.FloatTensor(trg_lens).to(target.device).requires_grad_()
        loss = losses.sum(dim=1)   # [batch_size]
        loss = loss / trg_lens_tensor
    else:
        loss = losses.sum(dim=1) # [batch_size]
    '''
    loss = losses.sum(dim=1)  # [batch_size]
    if orthogonal_loss:
        orthogonal_loss = compute_orthogonal_loss(delimiter_hidden_states, delimiter_hidden_states_lens)  # [batch_size]
        loss = loss + lambda_orthogonal * orthogonal_loss
    loss = loss.sum()

    # Debug
    if math.isnan(loss.item()):
        print("class distribution")
        print(class_dist)
        print("log dist flat")
        print(log_dist_flat)
        # raise ValueError("Loss is NaN")

    return loss


def compute_coverage_losses(coverage, attn_dist):
    """
    :param coverage: [batch_size, trg_seq_len, src_seq_len]
    :param attn_dist: [batch_size, trg_seq_len, src_seq_len]
    :return: coverage_losses: [batch, trg_seq_len]
    """
    batch_size = coverage.size(0)
    trg_seq_len = coverage.size(1)
    src_seq_len = attn_dist.size(2)
    coverage_flat = coverage.view(-1, src_seq_len)  # [batch_size * trg_seq_len, src_seq_len]
    attn_dist_flat = attn_dist.view(-1, src_seq_len)  # [batch_size * trg_seq_len, src_seq_len]
    coverage_losses_flat = torch.sum(torch.min(attn_dist_flat, coverage_flat), dim=1)  # [batch_size * trg_seq_len]
    coverage_losses = coverage_losses_flat.view(batch_size, trg_seq_len)  # [batch, trg_seq_len]
    return coverage_losses


def masked_coverage_loss(coverage, attn_dist, trg_mask):
    """
    :param coverage: [batch_size, trg_seq_len, src_seq_len]
    :param attn_dist: [batch_size, trg_seq_len, src_seq_len]
    :param trg_mask: [batch_size, trg_seq_len]
    :return:
    """
    src_seq_len = attn_dist.size(2)
    coverage_flat = coverage.view(-1, src_seq_len)  # [batch_size * trg_seq_len, src_seq_len]
    attn_dist_flat = attn_dist.view(-1, src_seq_len)  # [batch_size * trg_seq_len, src_seq_len]
    coverage_losses_flat = torch.sum(torch.min(attn_dist_flat, coverage_flat), 1)  # [batch_size * trg_seq_len]
    coverage_losses = coverage_losses_flat.view(*trg_mask.size())  # [batch, trg_seq_len]
    if trg_mask is not None:
        coverage_losses = coverage_losses * trg_mask
    return coverage_losses.sum()


def compute_orthogonal_loss(delimiter_hidden_states, delimiter_hidden_states_lens=None):
    """
    :param delimiter_hidden_states: [batch_size, decoder_size, max_num_delimiters]
    :return:
    """
    batch_size, decoder_size, max_num_delimiters = delimiter_hidden_states.size()
    identity = torch.eye(max_num_delimiters).unsqueeze(0).repeat(batch_size, 1, 1).to(
        delimiter_hidden_states.device)  # [batch, max_num_delimiters, max_num_delimiters]
    if delimiter_hidden_states_lens is not None:
        assert len(delimiter_hidden_states) == batch_size
        for i in range(batch_size):
            for j in range(max_num_delimiters - 1, delimiter_hidden_states_lens[i] - 1, -1):
                identity[i, j, j].fill_(0.0)
    orthogonal_loss_ = torch.bmm(torch.transpose(delimiter_hidden_states, 1, 2),
                                 delimiter_hidden_states) - identity  # [batch, num_delimiter, num_delimiter]
    orthogonal_loss = torch.norm(orthogonal_loss_.view(batch_size, -1), p=2, dim=1)  # [batch]
    return orthogonal_loss


"""
    :param class_dist: [batch_size, trg_seq_len, num_classes]
    :param target: [batch_size, trg_seq_len]
    :param trg_mask: [batch_size, trg_seq_len]
    :param divided_by_seq_len: boolean, whether to divide the loss by the max target sequence length
    :param trg_lens: a list with len of batch_size
    :param coverage_attn: boolean, whether to include coverage loss
    :param coverage: [batch_size, trg_seq_len, src_seq_len]
    :param attn_dist: [batch_size, trg_seq_len, src_seq_len]
    :param lambda_coverage: scalar, coefficient for coverage loss
    :return:
    """


def loss_debug():
    import torch.nn.functional as F
    import numpy as np
    torch.manual_seed(1234)
    np.random.seed(1234)

    num_classes = 5000
    batch_size = 5
    trg_seq_len = 6
    src_seq_len = 30
    class_dist = torch.randint(0, 5, (batch_size, trg_seq_len, num_classes))
    class_dist = F.softmax(class_dist, dim=-1)

    target = np.random.randint(2, 300, (batch_size, trg_seq_len))
    target[batch_size - 1, trg_seq_len - 1] = 0
    target[batch_size - 1, trg_seq_len - 2] = 0
    target[batch_size - 2, trg_seq_len - 1] = 0
    target = torch.LongTensor(target)

    trg_mask = np.ones((batch_size, trg_seq_len))
    target[batch_size - 1, trg_seq_len - 1] = 0
    target[batch_size - 1, trg_seq_len - 2] = 0
    target[batch_size - 2, trg_seq_len - 1] = 0
    trg_mask = torch.FloatTensor(trg_mask)

    divided_by_seq_len = True
    trg_lens = [trg_seq_len] * batch_size
    trg_lens[batch_size - 1] = trg_seq_len - 2
    trg_lens[batch_size - 2] = trg_seq_len - 1

    coverage_attn = True
    coverage = torch.rand((batch_size, trg_seq_len, src_seq_len)) * 5
    attn_dist = torch.randint(0, 5, (batch_size, trg_seq_len, src_seq_len))
    attn_dist = F.softmax(attn_dist, dim=-1)
    lambda_coverage = 1
    coverage_loss = True

    decoder_size = 100
    num_delimiter = 5
    delimiter_hidden_states = torch.randn(batch_size, decoder_size, num_delimiter)
    lambda_orthogonal = 0.03
    orthogonal_loss = True

    delimiter_hidden_states_lens = [3, 5, 2, 1, 5]

    loss = masked_cross_entropy(class_dist, target, trg_mask, trg_lens=trg_lens,
                                coverage_attn=coverage_attn, coverage=coverage, attn_dist=attn_dist,
                                lambda_coverage=lambda_coverage, coverage_loss=coverage_loss,
                                delimiter_hidden_states=delimiter_hidden_states, orthogonal_loss=orthogonal_loss,
                                lambda_orthogonal=lambda_orthogonal,
                                delimiter_hidden_states_lens=delimiter_hidden_states_lens)
    print(loss)
    return


def compute_orthogonal_loss_debug():
    import math

    batch_size_1 = 12
    decoder_size_1 = 100
    num_delimiter_1 = 5
    delimiter_hidden_states_1 = torch.randn(batch_size_1, decoder_size_1,
                                            num_delimiter_1)  # [batch_size, decoder_size, num_delimiter]
    ortho_loss_1 = compute_orthogonal_loss(delimiter_hidden_states_1)
    print(ortho_loss_1)
    assert ortho_loss_1.size() == torch.Size([batch_size_1])

    batch_size_2 = 2
    decoder_size_2 = 10
    num_delimiter_2 = 3
    delimiter_hidden_states_2 = torch.zeros(batch_size_2, decoder_size_2, num_delimiter_2)
    delimiter_hidden_states_2[0, 0, 0].fill_(1)
    delimiter_hidden_states_2[0, 1, 1].fill_(1)
    delimiter_hidden_states_2[0, 6, 2].fill_(1)
    delimiter_hidden_states_2[1, 5, 0].fill_(1)
    delimiter_hidden_states_2[1, 2, 1].fill_(1)
    delimiter_hidden_states_2[1, 2, 2].fill_(1)
    delimiter_hidden_states_2_lens = [3, 3]
    ortho_loss_2 = compute_orthogonal_loss(delimiter_hidden_states_2, delimiter_hidden_states_2_lens)
    # print(delimiter_hidden_states_2[0])
    print(ortho_loss_2)
    assert ortho_loss_2.size() == torch.Size([batch_size_2]) and ortho_loss_2[0].item() == 0.0 and math.fabs(
        ortho_loss_2[1].item() - math.sqrt(2)) < 1e-3

    batch_size_3 = 3
    decoder_size_3 = 10
    num_delimiter_3 = 4
    delimiter_hidden_states_3 = torch.zeros(batch_size_3, decoder_size_3, num_delimiter_3)
    delimiter_hidden_states_3[0, 0, 0].fill_(1)
    delimiter_hidden_states_3[0, 1, 1].fill_(1)
    delimiter_hidden_states_3[0, 6, 2].fill_(1)
    delimiter_hidden_states_3[0, 7, 3].fill_(1)
    delimiter_hidden_states_3[1, 5, 0].fill_(1)
    delimiter_hidden_states_3[1, 2, 1].fill_(1)
    delimiter_hidden_states_3[1, 2, 2].fill_(1)
    delimiter_hidden_states_3[2, 3, 0].fill_(1)
    delimiter_hidden_states_3[2, 3, 1].fill_(1)
    delimiter_hidden_states_3_lens = [4, 3, 2]
    ortho_loss_3 = compute_orthogonal_loss(delimiter_hidden_states_3, delimiter_hidden_states_3_lens)
    # print(delimiter_hidden_states_2[0])
    print(ortho_loss_3)
    assert ortho_loss_2.size() == torch.Size([batch_size_2]) and ortho_loss_2[0].item() == 0.0 and math.fabs(
        ortho_loss_2[1].item() - math.sqrt(2)) < 1e-3 and math.fabs(
        ortho_loss_3[1].item() - math.sqrt(2)) < 1e-3
    print("Pass!")


if __name__ == '__main__':
    compute_orthogonal_loss_debug()
    loss_debug()
