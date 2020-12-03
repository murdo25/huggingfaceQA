## SQuAD evaluation script. Modifed slightly for this notebook
from data_classes import ModelArguments
from transformers import HfArgumentParser
parser = HfArgumentParser(ModelArguments)


# Ben
# from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import os


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


model_args,  = parser.parse_json_file(json_file=os.path.abspath('args.json'))
print(model_args)
print(model_args.model_name_or_path)


# BEN
# model = T5ForConditionalGeneration.from_pretrained('models/tpu').to('cpu') # because its loaded on xla by default
# tokenizer = T5Tokenizer.from_pretrained('models/tpu')
model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
tokenizer = T5Tokenizer.from_pretrained(model_args.model_name_or_path)

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