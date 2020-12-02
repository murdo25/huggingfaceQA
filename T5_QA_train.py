# https://colab.research.google.com/github/patil-suraj/exploring-T5/blob/master/T5_on_TPU.ipynb#scrollTo=CaRw0ke1e1sF
# https://colab.research.google.com/github/patil-suraj/exploring-T5/blob/master/T5_on_TPU.ipynb

# T5 on TPU 💥🚀

# In this notebook we will see how to train T5 model on TPU with Huggingface's awesome new [trainer](https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py). We will train T5 base model on SQUAD dataset for QA task. We will use the recently released amazing [nlp](https://github.com/huggingface/nlp) package to load and process the dataset in just few lines.

# First make sure you are connected to the high RAM instance. This will not work on 12 GB colab instance.

# # Crash on purpose to get more ram :
# import torch
# torch.tensor([10.]*10000000000)
# 
# """Let's install [PyTorch/XLA](https://github.com/pytorch/xla) which enables PyTorch on TPU. Make sure you install the nightly version, as the trainer breaks on other versions."""
# 
# VERSION = "nightly"  #@param ["2.5" , "20200325", "nightly"]
# !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
# !python pytorch-xla-env-setup.py --version $VERSION
# 
# """Install transformers and the nlp package. Restart colab after this"""
# 
# !git clone https://github.com/huggingface/transformers.git
# !pip install ./transformers
# !pip install -U nlp

"""## Load and process data

Let's load and process the dataset using the nlp library. We will process the examples in follwoing way to cast QA task in text-to-text setting

**input**
question: question_text  context: context 

**target**
answer_text
"""

import torch
import nlp
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('t5-base')

# process the examples in input and target text format and the eos token at the end 
def add_eos_to_examples(example):
    example['input_text'] = 'question: %s  context: %s </s>' % (example['question'], example['context'])
    example['target_text'] = '%s </s>' % example['answers']['text'][0]
    return example

# tokenize the examples
def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], pad_to_max_length=True, max_length=512)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, max_length=16)

    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }

    return encodings

# load train and validation split of squad
train_dataset  = nlp.load_dataset('squad', split=nlp.Split.TRAIN)
valid_dataset = nlp.load_dataset('squad', split=nlp.Split.VALIDATION)

# map add_eos_to_examples function to the dataset example wise 
train_dataset = train_dataset.map(add_eos_to_examples)
# map convert_to_features batch wise
train_dataset = train_dataset.map(convert_to_features, batched=True)

valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)


# set the tensor type and the columns which the dataset should return
columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
train_dataset.set_format(type='torch', columns=columns)
valid_dataset.set_format(type='torch', columns=columns)

len(train_dataset), len(valid_dataset)

# cach the dataset, so we can load it directly for training

torch.save(train_dataset, 'train_data.pt')
torch.save(valid_dataset, 'valid_data.pt')

"""For more details on how to use the nlp library check out this [notebook](https://colab.research.google.com/github/huggingface/nlp/blob/master/notebooks/Overview.ipynb).

## Write training script

Using the `Trainer` is pretty straightforward. Here are the 4 basic steps which are needed to use trainer.

1. **Parse the arguments needed**. These are divided in 3 parts for clarity and seperation (TrainingArguments, ModelArguments and DataTrainingArguments).

  1. **TrainingArguments**: These are basicaly the training hyperparameters such as learning rate, batch size, weight decay, gradient accumulation steps etc. See all possible arguments [here](https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py). These are used by the Trainer.

  2. **ModelArguments**: These are the arguments for the model that you want to use such as the model_name_or_path, tokenizer_name etc. You'll need these to load the model and tokenizer.

  3. **DataTrainingArguments**: These are as the name suggests arguments needed for the dataset. Such as the directory name where your files are stored etc. You'll need these to load/process the dataset.

  TrainingArguments are already defined in the `TrainingArguments` class, you'll need to define `ModelArguments` and `DataTrainingArguments` classes for your task.




2. Load train and eval datasets
3. Initialize the `Trainer`

    These are the mininum parameters which you'll for initializing `Trainer`. For full list check [here](https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py#L107)

    ```
      model: PreTrainedModel
      args: TrainingArguments
      train_dataset: Optional[Dataset]
      eval_dataset: Optional[Dataset]
    ```
4. Start training with  `trainer.train`

    Call `trainer.train` and let the magic begin!


There are lots of things which the trainer handles for you out of the box such as gradient_accumulation, fp16 training, setting up the optimizer and scheduler, logging with wandb etc. I didn't set-up wandb for this experiment, but will explore it for sure in future experiment.
"""

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from transformers import T5ForConditionalGeneration, T5Tokenizer, EvalPrediction
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)


logger = logging.getLogger(__name__)



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # we will load the arguments from a json file, 
    #make sure you save the arguments in at ./args.json
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath('args.json'))
    print("\n\n")
    print("model args:", model_args)
    print("data args:", data_args)
    print("training args:", training_args)
    print("\n\n")
    exit()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer = T5Tokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    print('loading data')
    train_dataset  = torch.load(data_args.train_file_path)
    valid_dataset = torch.load(data_args.valid_file_path)
    print('loading done')

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=T2TDataCollator(),
        prediction_loss_only=True
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))
    
        results.update(eval_output)
    
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

"""## Train"""

import json

"""Let's write the arguments in a dict and store in a json file. The above code will load this file and parse the arguments."""

args_dict = {
  "num_cores": 8,
  'training_script': 'train_t5_squad.py',
  "model_name_or_path": 't5-base',
  "max_len": 512 ,
  "target_max_len": 16,
  "output_dir": './models/tpu',
  "overwrite_output_dir": True,
  "per_gpu_train_batch_size": 8,
  "per_gpu_eval_batch_size": 8,
  "gradient_accumulation_steps": 4,
  "learning_rate": 1e-4,
  "tpu_num_cores": 8,
  "num_train_epochs": 4,
  "do_train": True
}

with open('args.json', 'w') as f:
  json.dump(args_dict, f)

"""Start training!"""
# Ben
# import torch_xla.distributed.xla_multiprocessing as xmp
# xmp.spawn(_mp_fn, args=(), nprocs=8, start_method='fork')
main()

"""## Eval

There are two gotchas here. First the metrics functionality in the nlp package is still work-in-progress so we will use the official squad evaluation script. Second, for some reason which I couldn't figure out, the `.generate` method is not working on TPU so will need to do prediction on CPU. For predicting the validation set it almost takes 40 mins.
"""

## SQuAD evaluation script. Modifed slightly for this notebook

# Ben
# from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(gold_answers, predictions):
    f1 = exact_match = total = 0

    for ground_truths, prediction in zip(gold_answers, predictions):
      total += 1
      exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
      f1 += metric_max_over_ground_truths(
          f1_score, prediction, ground_truths)
    
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

import torch
# Ben
# import torch_xla
# import torch_xla.core.xla_model as xm

import nlp
from transformers import T5ForConditionalGeneration, T5Tokenizer

from tqdm.auto import tqdm

# model = T5ForConditionalGeneration.from_pretrained('models/tpu').to('cpu') # because its loaded on xla by default
# tokenizer = T5Tokenizer.from_pretrained('models/tpu')


model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath('args.json'))


# BEN
# model = T5ForConditionalGeneration.from_pretrained('models/tpu').to('cpu') # because its loaded on xla by default
# tokenizer = T5Tokenizer.from_pretrained('models/tpu')
model = T5ForConditionalGeneration.from_pretrained("t5-large")
tokenizer = T5Tokenizer.from_pretrained("t5-large")

valid_dataset = torch.load('valid_data.pt')
dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=2)




answers = []
for batch in tqdm(dataloader):
  print("batch", type(batch))
  outs = model.generate(input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'],
                        max_length=16,
                        early_stopping=True)
  outs = [tokenizer.decode(ids) for ids in outs]
  answers.extend(outs)
  break

predictions = []
references = []

def clean(result):
    result = result.replace("<pad>","")
    result = result.replace("</s>", "")
    result = result.strip()
    return result


result = tokenizer.decode(valid_dataset[0]['target_ids'])
print("target:", clean(answers[0]), "output:", clean(result))
if(clean(answers[0]) == clean(result)):
    print("worked")

# for ref, pred in zip(valid_dataset, answers):
#   print("ref", ref)
#   print("ref.keys()", ref.keys())
#   predictions.append(pred)
#   references.append(ref['answers']['text'])
#   break

# print(predictions[0], references[0])

# you might want to evaluate that...
# evaluate(references, predictions)
# evaluate(references, predictions)
# print("eval:", evaluate(references, predictions))
print("eval:", evaluate(answers[0], clean(result)))
print("finished!")