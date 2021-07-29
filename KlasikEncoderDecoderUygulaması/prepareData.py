import string
from os import listdir
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import load_img,img_to_array
from keras.models import Model
from pickle import dump



          ## --- PREPARE REPORT DATA --- ##
def clean_report(img_id_all,reports_dictionary):
  '''
  bu fonksiyon yeni bir dict yapısı oluşturur. Şöyle ki yukarıdaki incelemeye göre mevcut img idlere göre, olmayanalrın raporlarını atar.
  girdiler : tüm img id'ler ve başlangıçta oluşturulan rapor dict yapısı
  çıktı : mevcut raporların dict yapısı
  '''
  new_reports_dictionary = dict()
  for img_id,finding in reports_dictionary.items():
    if img_id in img_id_all:
      new_reports_dictionary[img_id] = finding
  print("Dict's element count:",len(new_reports_dictionary))
  return new_reports_dictionary

def get_findings(reports_dictionary):
  '''
  Sadece raporlardaki findings'leri bir değişkene almak ya da dosyaya kaydetmek için kullanalabilirim.
  girdi : raporların olduğu en güncel dict yapısı
  çıktı : tüm findings olduğu bir liste
  '''
  findings = list()
  for img_id in reports_dictionary.keys():
    finding = reports_dictionary[img_id]
    findings.append(finding)
  print("Findings count:",len(findings))
  return findings

def clean_findings(findings_all):
  '''
  Tüm text dosyasını alır. Bizim için findingsler ve onları temizleyip yeni hallerine güncellememi sağlayan fonksiyondur.
  '''
  findings_all = [word.lower() for word in findings_all] # kücük harf yapar.
  findings_all = [word.translate(str.maketrans('','',string.punctuation)) for word in findings_all] # noktalama işaretlerini siler. String.puncuation bize tüm noktalama işaretlerinin olduğu listeyi verir.
  print("Clean findings count:",len(findings_all))
  return findings_all

            ###   ---   PREPARE IMAGE DATA   ---    ##
          #### FEATURE EXTRACTION AND SAVING THESE  ####

# Feature extraction from images. Son 2 katman zaten classification için kullanılıyor ona ihtiyaç yok.Bize lazım olan özellik vektörü.
# Save features as ".pkl" file
def extract_features(images_path):
  '''
  Bu fonksiyon görüntü verilerinden feature çıkarıyor.
  Input : görüntünün dosya yolu, nerede olduğu.
  output : görüntü image id sinin ve feature'ının olduğu dict yapısı döndürür.
  '''
  model = VGG19()
  # restructure model
  model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
  print(model.summary())

  features = dict()  # image feature vektörünü ve image idleri tutacagım liste.
  step = 0  # hücre çalışırken naptıgını görebilmek için ekledim.
  for filename in listdir(images_path):  # tüm imageleri dolaşan loop
    img_path = images_path + '/' + filename
    image = load_img(img_path, target_size=(224, 224))  # Aldığım görüntüyü yeniden boyutlandırdım.
    image = img_to_array(image)  # Görüntünün piksel değerlerini array haline getirdim.
    w, h, ch = image.shape[0], image.shape[1], image.shape[
      2]  # Image'in Weight, Height, Channel değerleri (Weightler değişiyor,height=512,ch=3)
    image = image.reshape((1, w, h, ch))  # her seferinde 1 resim aldığım için 1 ,w,h,ch.

    # preprocess, prepare image for VGG model
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)  # VGG19 modeline göre feature prediction yaptım.
    image_id = filename.split(sep='.')[
      0]  # dosya isimleri .png ile bittiğinden '.' ya göre seperate edip bir array oluşturdum. İlk elemanı bizim image-idmiz olduğundan ilk indeksteki elemanı aldım.

    features[image_id] = feature  # image id ve feature değerlerini eşleştirdim.
    step += 1  # adım sayısını arttırdım.
    print(f"{step}. step")  # kacıncı adımda oldugumu görmek için adımı yazdırdım.
    print(
      '>%s ;extracting features..' % filename)  # hangi görüntü üzerinde özellik çıkarımı yaptığımı görmek için yazdırdım.
  return features


def save_as_pkl(features, filename):
  '''
  Bu fonksiyon tekrar tekrar aynı adımı gerçekleştirmeyeyim diye "extract_features()" fonksiyonundan gelen "features" değişkenini kaydediyor.
  '''
  try:
    dump(features, open(filename, 'wb'))
    print(f"{filename} is saved")
  except:
    raise Exception(f'{filename} can not be saved..!')


