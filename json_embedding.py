import csv
import subprocess
import torch

#command = r'''python code\get_embedding.py -s 0.json -u "pre-trained_model\UI2Vec_model.ep120" -m 
#"pre-trained_model\Screen2Vec_model_v4.ep120" -l "pre-trained_model\layout_encoder.ep800"'''

output = subprocess.check_output(args=['python', 'code\get_embedding.py', '-s', '0.json', '-u', r'pre-trained_model\UI2Vec_model.ep120', '-m', 
r'pre-trained_model\Screen2Vec_model_v4.ep120', '-l', r'pre-trained_model\layout_encoder.ep800'], shell=True, stderr=subprocess.STDOUT)
print(output)
csv_file = 'json_vectors.csv'

#try:
#    with open(csv_file, 'w') as csvfile:
#        writer = csv.Dict
