import generateReport as RG
import radgorTrain as RT
from keras.models import load_model
from nltk import corpus_bleu
# evaluate the skill of the model

def evaluate_model(model, findings_map, features, tokenizer, max_length):
  print("Evaluate model çalışıyor..")
  y_all, yhat_all = list(), list()
  # step over the whole set
  i = 1 #step count
  for img_id, finding in findings_map.items():
    yhat = RG.generate_desc(model, tokenizer, features[img_id], max_length)
    print(f"{i}. step : generating 'findings' for: {img_id}")
    i +=1
    # store actual and predicted
    references = finding.split()
    print(f"FINDING: {finding}, REFERENCES:{references}")
    y_all.append(references)
    yhat_all.append(yhat.split())
  return y_all,yhat_all

def evaluation_BLEU_metric(y_all,yhat_all):
  print('BLEU-1: %f' % corpus_bleu(y_all, yhat_all, weights=(1.0, 0, 0, 0)))
  print('BLEU-2: %f' % corpus_bleu(y_all, yhat_all, weights=(0.5, 0.5, 0, 0)))
  print('BLEU-3: %f' % corpus_bleu(y_all, yhat_all, weights=(0.3, 0.3, 0.3, 0)))
  print('BLEU-4: %f' % corpus_bleu(y_all, yhat_all, weights=(0.25, 0.25, 0.25, 0.25)))

my_vgg19_model_filename = 'C:/Users/umtsr/Desktop/BM495/Uygulama Dosyaları/models/model_loss2.464-val_loss2.533.h5'
my_vgg19_model = load_model(my_vgg19_model_filename)

y,yhat = evaluate_model(my_vgg19_model,RT.dev_findings_map,RT.dev_features,RT.tokenizer_findings,RT.max_length)
evaluation_BLEU_metric(y,yhat)