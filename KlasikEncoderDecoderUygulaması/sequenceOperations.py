from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

def create_vocab(findings_all):
  '''
  Bu fonksiyon sözlük oluşturur.
  Girdi : en temiz findings'leri yolla
  çıktı : tüm eşsiz kelimeleri içeren list.
  '''
  vocab = set()
  for finding in findings_all:
    for word in finding.split(): #findingsleri boşluk karakterine göre ayırarak kelimeleri elde ettim.
      vocab.add(word) # kelimeleri sözlüğe ekledim. Set yapısı olduğu için aynı kelimeleri eklemez.
  print("Unique word(vocab) num:",len(vocab))
  return vocab

# fit_on_texts de sözlük oluşturuyor.
def createTokenizer(findings_all_clean):
  '''
  fit_on_text sözlük oluşturmak için kullanılan alternatif bir yol tokenizer sınıfından faydalanılarak kolayca halledilebiliyor.
  '''
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(
    findings_all_clean)  # text'i kendisinin kullanabileceği formata çeviriyor. İçeriği yazdırmayı denedim ama göremedim.
  return tokenizer


def vektorization(tokenizer):
  '''
  Sözlükteki kelimeleri bir dict yapısı halinde indekslerle eşleştiren fonksiyon.
  Girdi : tokenizer.fit_on_texts(text_verisi)
  çıktı: sözlük.

  '''
  word2index = tokenizer.word_index
  index2word = tokenizer.index_word
  return word2index, index2word


def findMax(findings_all_clean):
  '''
  Bu fonksiyon en fazla kelimenin kullanıldığı sequence'i hesaplar ve döndürür. Böylece sequence oluştururlurken eksik bir sequence olmaz.
  Input : tüm findingsleri alır
  output: tüm findingler içerisinde en uzun findingin kelime sayısını döndürür.
  '''
  maxLen = 0 # başlangıç değeri
  for f in findings_all_clean: # tüm findings dolaş
    words = f.split() # kelimelere ayır
    count = len(words) # kelimelere ayırdığında liste haline geldiği için len ile kelime sayısı hesaplanabilir.
    if count > maxLen: # max kelime sayısını güncelleyen if
      maxLen = count
  print("Maximum length:",maxLen)
  return maxLen


def create_sequences(tokenizer, max_length, findings_map, features_map, vocab_size):
  '''
  Bu fonksiyon sequence oluşturur. Tokenizer bizim yazıyı(findings) işlenebilir hale getirdiğimiz nesne.
  Input: tokenizer : findingslerden oluşan token
         max_length: en uzun sequence sayısı
         findings_map : tüm findingsler ve image_id lerin ilişkilendirildiği dict yapısı
         features_map : tüm image featureların ve image_idlerin ilişkilendirildği dict yapısı
         vocab_size : Sözlük boyutu
  Output:
      Ximages = tüm imagelerin feature'ı ( Boyut : toplam feature sayısı, feature matrisi boyutu)
      Xsequence = tüm sequenceler     ( Boyut : toplam sequence sayısı, sequence uzunlugu)
      y_out = sonraki kelime      ( Boyut : toplam sonraki kelime sayısı, sözlükteki kelime sayısı(cunku sonraki kelim bu sözlük içerisindeki kelimelerden biridir.))
  Kücük not : tokenizer nesnesinden çalıştırılan fonksiyonların çıktısı hep 2-dimensional array ve 0. indeksteki değerler bizim işimize yarayacak olan değerlerdir.
  '''
  Ximages, Xsequence, y_out = list(), list(), list()
  for img_id, finding in findings_map.items():
    # Sequence oluşturuluyor,encoding.
    sequence = tokenizer.texts_to_sequences([finding])[0]
    # Sequence üzerinde dolaşıyoruz.
    for i in range(1, len(sequence)):  # Bir sequence içerisindeki tüm kelimeleri dolaşan lopp
      # Son kelime hariç tüm sequence parça parça alıyoruz her loop da
      in_seq = sequence[:i]
      # her loop sonundaki son kelime
      out_seq = sequence[i]
      # padding ekliyoruz
      in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
      # one hot encoding uyguluyoruz.
      out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
      # sequence'in eşleştiği resmin featurelarını alıyoruz
      img_feature = features_map[img_id][0]
      # ekleme işlemleri yapılıyor.yedekle.
      Ximages.append(img_feature)
      Xsequence.append(in_seq)
      y_out.append(out_seq)

  return np.array(Ximages), np.array(Xsequence), np.array(y_out)

