# Frozen hyperparameter policy (pre-registered)
# One recipe selected on E1's validation slice, frozen across ALL arms so arms
# differ only in inputs and pretraining, not in tuning:
#   backbone resnet18, AdamW lr 1e-3, weight_decay 1e-4, cosine schedule,
#   batch 64, max 30 epochs, early stop patience 7 on val macro-F1,
#   augmentation flips + 90-deg rotations, input 64x64.
# An arm-specific learning-rate check on E2's val is left to future work.
