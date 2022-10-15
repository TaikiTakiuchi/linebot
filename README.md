---
license: mit
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: output
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# output

This model is a fine-tuned version of [rinna/japanese-gpt2-small](https://huggingface.co/rinna/japanese-gpt2-small) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0834
- Accuracy: 0.9857

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 100.0

### Training results



### Framework versions

- Transformers 4.24.0.dev0
- Pytorch 1.12.1+cu113
- Datasets 2.5.2
- Tokenizers 0.13.1
