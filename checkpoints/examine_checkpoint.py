import sys
import json

if len(sys.argv) < 2:
  sys.exit('Usage: python examine_checkpoint.py model_id*.json')

checkpoint = json.load(open(sys.argv[1]))
acc_history = checkpoint['val_accuracy_history']
acc_history = dict((int(x), acc_history[x]) for x in acc_history)
acc_key = sorted([x for x in acc_history])
for k in acc_key:
  print acc_history[k]

print
print 'best accuracy:', max(acc_history.values())
