# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.utils import orthogonal_regularization


class PPO_DICE():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 discriminator,
                 div_type,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 gamma,
                 lambda_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=False,
                 ortho_norm=False,
                 disc_train=1,
                 disc_lr_factor=1,
                 discrete_actions=False,
                 gradient_penalty=False):

        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.discriminator = discriminator
        self.gamma = gamma
        self.lambda_coef = lambda_coef
        self.div_type = div_type
        self.discrete_actions = discrete_actions

        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.ortho_norm = ortho_norm
        self.gradient_penalty = gradient_penalty
        self.use_clipped_value_loss = use_clipped_value_loss
        self.disc_train = disc_train
        self.disc_lr_factor = disc_lr_factor

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.disc_optimizer = optim.Adam(discriminator.parameters(), lr=disc_lr_factor * lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        disc_loss_epoch = 0
        max_rat_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:  # not supporting this currently
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch, next_obs=True)

            for sample in data_generator:
                init_obs_batch, obs_batch, recurrent_hidden_states_batch, actions_batch, next_obs_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                   adv_targ = sample

                # treat each obs as initial obs
                init_obs_batch = obs_batch

                _, init_action_batch, init_actions_log_probs, _ = self.actor_critic.act(
                    init_obs_batch, None, torch.ones_like(masks_batch), reparam=not self.discrete_actions)

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, None, masks_batch,
                    actions_batch)

                # compute a' -- should this be done without gradient?
                _, next_actions_batch, next_actions_log_probs, _ = self.actor_critic.act(
                    next_obs_batch, None, masks_batch, reparam=not self.discrete_actions)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                # surr = ratio * adv_targ
                # action_loss = -surr.mean()

                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                def compute_disc_loss(discriminator, reinforce=False):
                    g_init = discriminator(init_obs_batch, init_action_batch.float())
                    g = discriminator(obs_batch, actions_batch)
                    g_next = discriminator(next_obs_batch, next_actions_batch.float())
                    bell_residuel = g - self.gamma * g_next

                    if self.div_type == 'ChiS':
                        disc_linear_loss = (1 - self.gamma) * g_init
                        disc_nonlinear_loss = bell_residuel + 0.25 * bell_residuel**2
                        disc_loss = (disc_nonlinear_loss - disc_linear_loss).mean()
                    elif self.div_type == 'KL':
                        disc_linear_loss = (1 - self.gamma) * g_init
                        if reinforce:
                            disc_nonlinear_loss = -self.gamma * torch.nn.functional.softmax(
                                bell_residuel, dim=0).detach() * g_next.detach()
                            disc_loss = (next_actions_log_probs * disc_nonlinear_loss).sum() - (disc_linear_loss.detach() * init_actions_log_probs).mean()
                        else:
                            disc_nonlinear_loss = torch.nn.functional.softmax(bell_residuel, dim=0).detach() * bell_residuel
                            disc_loss = disc_nonlinear_loss.sum() - disc_linear_loss.mean()
                    else:
                        AssertionError('{self.div_type} not supported.'.format(self.div_type))

                    return disc_loss

                def compute_gradient_penaly(discriminator):
                    obs_inputs = torch.autograd.Variable(torch.cat([obs_batch, next_obs_batch], dim=0),
                                                         requires_grad=True)
                    action_inputs = torch.autograd.Variable(torch.cat([actions_batch.float(), next_actions_batch.float()], dim=0),
                                                            requires_grad=True)
                    disc_outputs = discriminator(obs_inputs, action_inputs)

                    gradient = torch.autograd.grad(outputs=disc_outputs, inputs=[obs_inputs, action_inputs],
                                                   grad_outputs=torch.ones(disc_outputs.size(), device=self.device),
                                                   create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)

                    gradient = torch.cat(gradient, dim=1)
                    gradient = gradient.view(gradient.size(0), -1)
                    diff = gradient.norm(2, dim=1) - 1
                    return torch.mean(diff**2)



                ##################
                # train discriminator
                ##################
                for _ in range(self.disc_train):
                    disc_loss = compute_disc_loss(self.discriminator)
                    if self.ortho_norm:
                        disc_loss += 1e-4 * orthogonal_regularization(self.discriminator, self.device).squeeze()
                    if self.gradient_penalty:
                        disc_loss += 10 * compute_gradient_penaly(self.discriminator)

                    self.disc_optimizer.zero_grad()
                    disc_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.discriminator.parameters(),
                                             self.max_grad_norm)
                    self.disc_optimizer.step()

                ###############
                # train policy
                ###############
                disc_loss = compute_disc_loss(self.discriminator, reinforce=self.discrete_actions)

                # get 90th percentile
                n = adv_targ.shape[0]
                percentile_idx = n // 10
                lambda_coef = torch.sort(torch.abs(adv_targ), descending=True)[0][percentile_idx]

                self.optimizer.zero_grad()
                actor_critic_loss = value_loss * self.value_loss_coef + action_loss - \
                                    dist_entropy * self.entropy_coef - lambda_coef.detach() * disc_loss
                # if self.ortho_norm:
                #     actor_critic_loss += 1e-4 * orthogonal_regularization(self.actor_critic, self.device).squeeze()

                actor_critic_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    values, new_action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                    ratios = torch.exp(new_action_log_probs - action_log_probs)
                    max_rat = ratios.max()

                max_rat_epoch = max(max_rat_epoch, max_rat.item())
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                disc_loss_epoch += disc_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        disc_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, disc_loss_epoch, max_rat_epoch
