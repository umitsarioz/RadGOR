from os import listdir
from pickle import load
import csv


def get_images_filenames(path):
  '''
  fonksiyon tüm radyolojik görüntülerin dosya isimlerini alır ayırır, kısaca tüm image id leri alıp döndüren fonksiyon
  girdi : görüntülerin olduğu klasör yolu
  çıktı : list
  '''
  filenames = list()
  for filename in listdir(path):
    if filename.endswith('.png'):
      filenames.append(filename.split('.')[0])
  print("Images File names got.File count:",len(filenames))
  return filenames

def load_report(path):
  '''
  Bu fonksiyon tüm raporların içindeki image id ve finding değerlerini döndürür.
  girdi : raporların olduğu dosyanın yolu
  çıktı : findings ve img id leri içeren dict yapısı döndürür.
  '''
  reports_dictionary = dict()
  with open(path,'r') as file:
    reader = csv.reader(file,delimiter=',')
    for line in reader:
      img_id = line[0]
      finding = line[1]
      reports_dictionary[img_id] = finding
  print("Dict's elemnt count:",len(reports_dictionary))
  return reports_dictionary

def load_features(feature_file,images_id_all):
  '''
  Bu fonksiyon daha önceden çıkarılmış feature'ları yükler.
  input : feature dosyasını ve tüm resimlerin image_idlerini alır.
  output : Tüm image_id değerlerini features dosyası içinde arayarak bulur ve bunu bir dict yapısına kaydeder. Yukarıda oluşturduğumuz yapıyı tekrar oluşturur.
  '''
  features = load(open(feature_file,'rb'))
  features_dict = dict()
  for img_id in images_id_all:
      features_dict[img_id] = features[img_id]
  print("Feature is loading.. Feature count:",len(features_dict))
  return features_dict

def load_findings_all_clean_dict(findings_all_clean,images_id_all):
  '''
  Bu fonksiyonla sequence oluşturmadan önce findings kısmına eklemeler yapıldı. Bu sayede sequence'in başlangıc ve bitişi rahatça sağlanabilir.
  Input : tüm temizlenmiş(kücükharf/noktalama işaretsiz) ve tüm image id leri alır.
  çıktı : başlangıç ve bitiş kelimesi verilmiş yeni sequenceleri oluşturur.
  '''
  findings_dict = dict()
  for i in range(len(findings_all_clean)):
    finding = 'startseq ' + findings_all_clean[i] + ' endseq' # Evaluation ya da prediction yaparken cümle başlangıcı ve bitişinde kullanılır.
    img_id = images_id_all[i]
    findings_dict[img_id] = finding
  return findings_dict

