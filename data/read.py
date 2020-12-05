import json
import random


# data_file='mini.jsonl'
# data_file='first100.jsonl'
data_file='1k.jsonl'


with open(data_file, 'r') as json_file:
    json_list = list(json_file)

json_lines = []
for json_str in json_list:
    json_lines.append(json.loads(json_str))


valid_questions=[]
for l in json_lines:
    # print(l['annotations'][0]['short_answers'])
    # clear all docs with more or less than one answer
    if(len(l['annotations'][0]['short_answers']) == 1):
        valid_questions.append(l)

print('num valid:', len(valid_questions))


def get_exert(doc, start_token, end_token):
    return doc.split(' ')[start_token:end_token]

def get_short_answer(q):
    answer_indx = q['annotations'][0]['short_answers'][0]
    return get_exert(q['document_text'], answer_indx['start_token'], answer_indx['end_token'])

def get_long_answer(q):
    # print('annotations:', q['annotations'])
    answer_indx = q['annotations'][0]['long_answer']
    # print(answer_indx)
    # print('long answer correct index:', answer_indx)
    return get_exert(q['document_text'], answer_indx['start_token'], answer_indx['end_token'])

def get_random_negative(q):
    long_answer_indx = q['annotations'][0]['long_answer']

    for i in range(len(q['long_answer_candidates'])):
        if(q['long_answer_candidates'][i]['start_token'] == long_answer_indx['start_token']):
            del q['long_answer_candidates'][i]
            break

    # print(q['long_answer_candidates'][0])
    # q['long_answer_candidates'].remove(long_answer_indx)
    answer_indx = random.choice(q['long_answer_candidates'])
    return get_exert(q['document_text'], answer_indx['start_token'], answer_indx['end_token'])


q = valid_questions[0]
print(q.keys())
print('\n\n')
print(q['question_text'])
print(' '.join(get_long_answer(q)))
print(' '.join(get_short_answer(q)))
print('\n\n')
print('negative:', ' '.join(get_random_negative(q)))

class Datapoint:
    def __init__(self):
        self.text = None
        self.question = None
        self.target = None

positive_datapoints = []
negitave_datapoints = []
for q in valid_questions:
    d = Datapoint()
    # Construct positive example 
    d.question = q['question_text']
    d.target = get_short_answer(q)
    d.text = get_long_answer(q)
    positive_datapoints.append(d)
    # Construct negitive example 
    d.text = get_random_negative(q)
    d.target = "None"
    negitave_datapoints.append(d)


print("total positive examples:", len(positive_datapoints))
print("total negitive examples:", len(negitave_datapoints))