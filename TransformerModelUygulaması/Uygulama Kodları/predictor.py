import tensorflow as tf
from pickle import load
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
import os

def goruntuYukle(image_path):
   img = tf.io.read_file(image_path)
   img = tf.image.decode_png(img, channels=3)
   img = tf.image.resize(img, (299, 299))
   img = preprocess_input(img)
   return img

def create_padding_mask(seq):
   seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
   return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
   mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
   return mask  # (seq_len, seq_len)

def create_masks_decoder(tar):
   look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
   dec_target_padding_mask = create_padding_mask(tar)
   combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
   return combined_mask


def generateOneCaption(image_path,extraction_model,prediction_model,tokenizer,SEQ_LENGTH):
   temp_input = tf.expand_dims(goruntuYukle(image_path), 0)
   img_tensor_val = extraction_model(temp_input)
   img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
   start_token = tokenizer.word_index['<start>']
   end_token = tokenizer.word_index['<end>']
   decoder_input = [start_token]

   output = tf.expand_dims(decoder_input, 0) #tokens
   predicted_words = [] #word list
   #print("Inıtial: Outpu:",output,"\nDeocder_inp:",decoder_input,"\nstart_token:",start_token)
   for i in range(SEQ_LENGTH):
      dec_mask = create_masks_decoder(output)
      predictions = prediction_model(img_tensor_val,output,False,dec_mask)
      predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
      predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
      if predicted_id == end_token:
         predicted_caption = [' '.join(pred_word for pred_word in predicted_words)]
         return predicted_caption[0] # tek elemanlı liste
      predicted_words.append(tokenizer.index_word[int(predicted_id)])
      output = tf.concat([output, predicted_id], axis=-1)
      #print("output:",output,"\noutput.shape:",output.shape)
      #print("predicted_id:",predicted_id)
   predicted_caption = [' '.join(pred_word for pred_word in result)]
   return predicted_caption[0] # tek elemanlı liste