#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/10/6 0:02
# @Author : WeiHua

import torch
from torch.nn import Module
from .bregman_pytorch import sinkhorn


# code inherited from "https://github.com/cvlab-stonybrook/DM-Count/blob/master/losses/ot_loss.py"
class OT_Loss(Module):
    def __init__(self, num_of_iter_in_ot=100, reg=10.0, method='sinkhorn'):
        super(OT_Loss, self).__init__()
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg
        self.method = method

    def forward(self, t_scores, s_scores, pts, cost_type='all', clamp_ot=False, aux_cost=None):
        """
        Calculating OT loss between teacher and student's distribution.
        Cost map is defined as: cost = dist(p_t, p_s) + dist(score_t, score_s).
        All dist are l2 distance.
        Args:
            t_scores: Tensor with shape (N, )
            s_scores: Tensor with shape (N, )

        Returns:

        """
        assert cost_type in ['all', 'dist', 'score']
        with torch.no_grad():
            t_scores_prob = torch.softmax(t_scores, dim=0)
            s_scores_prob = torch.softmax(s_scores, dim=0)
            score_cost = (t_scores.detach().unsqueeze(1) - s_scores.detach().unsqueeze(0)) ** 2
            score_cost = score_cost / score_cost.max()
            if cost_type in ['all', 'dist']:
                coord_x = pts[:, 0]
                coord_y = pts[:, 1]
                dist_x = (coord_x.reshape(1, -1) - coord_x.reshape(-1, 1)) ** 2
                dist_y = (coord_y.reshape(1, -1) - coord_y.reshape(-1, 1)) ** 2
                dist_cost = (dist_x + dist_y).to(t_scores_prob.device)
                dist_cost = dist_cost / dist_cost.max()
                if cost_type == 'all':
                    cost_map = dist_cost + score_cost
                else:
                    cost_map = dist_cost
            else:
                cost_map = score_cost
            if not isinstance(aux_cost, type(None)):
                cost_map = cost_map + aux_cost
            # cost_map = (dist_cost + score_cost) / 2
            source_prob = s_scores_prob.detach().view(-1)
            target_prob = t_scores_prob.detach().view(-1)
            if t_scores.shape[0] < 2000: # 2500
                _, log = sinkhorn(target_prob, source_prob, cost_map, self.reg,
                                  maxIter=self.num_of_iter_in_ot, log=True,
                                  method=self.method)
                beta = log['beta']  # size is the same as source_prob: [#cood * #cood]
            else:
                _, log = sinkhorn(target_prob.cpu(), source_prob.cpu(), cost_map.cpu(), self.reg,
                                  maxIter=self.num_of_iter_in_ot, log=True,
                                  method=self.method)
                beta = log['beta'].to(target_prob.device)  # size is the same as source_prob: [#cood * #cood]
        # compute the gradient of OT loss to predicted density (unnormed_density).
        # im_grad = beta / source_count - < beta, source_density> / (source_count)^2
        source_density = s_scores.detach().view(-1)
        source_count = source_density.sum()
        im_grad_1 = (source_count) / (source_count * source_count + 1e-8) * beta  # size of [#cood * #cood]
        im_grad_2 = (source_density * beta).sum() / (source_count * source_count + 1e-8)  # size of 1
        im_grad = im_grad_1 - im_grad_2
        im_grad = im_grad.detach()
        # Define loss = <im_grad, predicted density>. The gradient of loss w.r.t prediced density is im_grad.
        if clamp_ot:
            return torch.clamp_min(torch.sum(s_scores * im_grad), 0)
        return torch.sum(s_scores * im_grad)
