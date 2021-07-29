import numpy as np
from keras.models import Model
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.sequence import pad_sequences


def word_for_id(integer,tokenizer):
  for word, index in tokenizer.word_index.items():
    if index == integer:
      return word
  return None

# generate a report for an image
def generate_desc(model, tokenizer, imageFeatures, max_length):
  print("Generate description methodu çalışıyor..")
  # seed the generation process
  in_text = '<start>'
  # iterate over the whole length of the sequence
  for i in range(max_length):
    # integer encode input sequence
    sequence = tokenizer.texts_to_sequences([in_text])[0]
    # pad input
    sequence = pad_sequences([sequence], maxlen=max_length)
    # predict next word
    try:
      yhat = model.predict([imageFeatures,sequence])
      print(f"{i}. prediction:",yhat)
    except Exception as e:
      print("EXCEPTION:",e)
    # convert probability to integer
    yhat = np.argmax(yhat)
    # map integer to word
    word = word_for_id(yhat, tokenizer)
    # stop if we cannot map the word
    if word is None:
      break
    # append as input for generating the next word
    in_text += ' ' + word
    # stop if we predict the end of the sequence
    if word == '<end>' or word == 'end':
      break

  try:
    print("Silme block çalıştı..")
    print("My intext:",in_text)
    in_text  = in_text.replace('<start>','')
    in_text = in_text.replace('<end>','.')
    in_text = in_text.replace('end','.')
  except:
    pass

  return in_text


def extract_features_new_image(filename):
    # load the model
    model = VGG19()
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # load the photo
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0],image.shape[1],image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image)
    return feature
