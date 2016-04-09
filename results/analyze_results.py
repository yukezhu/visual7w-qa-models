from __future__ import print_function
import sys
import json

if len(sys.argv) < 2:
  sys.exit('Usage: python analyze_results.py result*.json')

# load result json
results = json.load(open(sys.argv[1]))
predictions = results['split_predictions']

# 7W question types
types = ['what', 'where', 'when', 'who', 'why', 'how', 'which']

# get question type by first word
def get_type(question):
  for k, t in enumerate(types):
    if question.startswith(t):
      return k
  return None

# count statistics
total_n = dict()
correct_n = dict()
for p in predictions:
  q_type = get_type(p['question'])
  if q_type is not None:
    total_n[q_type] = total_n.get(q_type, 0) + 1
    if p['selected'] == 1:
      correct_n[q_type] = correct_n.get(q_type, 0) + 1

# count total number of QA
num_correct = sum(correct_n.values())
num_total = sum(total_n.values())

# print question types
for k, t in enumerate(types):
  if k in total_n:
    print(t, end='\t| ')
print('overall')

print('-'*58)

# print accuracy
for k, t in enumerate(types):
  if k in total_n:
    print('%.3f' % (1.0 * correct_n[k] / total_n[k]), end='\t| ')
print('%.3f' % (1.0 * num_correct / num_total))
