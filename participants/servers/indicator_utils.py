"""Shared indicator objective and configuration validation."""

import math

import torch


def parameter_distance(model, reference):
    differences = [
        (parameter - reference[name].detach()).reshape(-1)
        for name, parameter in model.named_parameters()
    ]
    return torch.linalg.vector_norm(torch.cat(differences), ord=2)


def validate_indicator_schema(params, implementation):
    legacy = {'indicator_dataset_size', 'indicator_batch_size', 'indicator_ood_type',
              'indicator_train_iterations', 'indicator_train_epochs', 'indicator_lr',
              'indicator_momentum', 'indicator_weight_decay', 'indicator_l2_lambda',
              'indicator_reject_threshold'}
    modern = {'indicator_size', 'indicator_plant_steps', 'indicator_plant_lr',
              'indicator_plant_momentum', 'indicator_plant_batch_size', 'indicator_prox_lambda',
              'indicator_threshold', 'indicator_ood_source', 'indicator_eval_batch_size',
              'indicator_fallback_keep_top'}
    wrong = legacy if implementation == 'indicator' else modern
    supplied = sorted(wrong.intersection(params))
    if supplied:
        raise ValueError(f'{implementation} received settings for the other indicator implementation: {supplied}')


def validate_coefficient(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Indicator regularization coefficient must be finite and nonnegative')
    return value
