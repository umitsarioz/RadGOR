

from xml.etree import ElementTree as ET
import os
import csv
import pandas as pd

def xmlRaporuAl(report_name):
    tree = ET.parse(report_name)
    root = tree.getroot()
    return root

# Id sayısı kadar rapor olusturacağım için önce kaç Id olduğunu buluyorum.
def id_Bul():
    '''
    Bu fonksiyon raporların ilişkili olduğu görüntünün id sini bulmamıza yardımcı olur.
    '''
    my_Images = []  # Goruntuleri tutacak olan liste
    for img in root.getchildren():  # xml dosyası içinde ve root altındaki tüm tag'leri dolaş
        if img.tag == 'parentImage':  # bu tagler'den parentImage eşit ise
            img_id = list(img.attrib.values())[
                0]  # attribute degerlerinin ilk elemanı bize raporun ait oldugu image id donduruyor
            my_Images.append(img_id)  # bunu id'ler icin olusturdugum listeye ekle
    return my_Images


def finding_Bul():
    '''
    Bu fonksiyon xml tipindeki raporlar içerisindeki findings kısmını bulmamıza yardımcı olur.
    '''
    findings = root.find('./MedlineCitation/Article/Abstract/AbstractText[@Label="FINDINGS"]')

    finding_text = findings.text
    if finding_text == None:
        finding_text = 'No Findings'
    return finding_text

def satirEkle(report_id, finding):
    '''
    Bu fonksiyon bize sadece id ve findinglerin oldugu csv dosyasını oluşturmamızda yardımcı olur.
    inputs : report_id - raporumuzun ilişkili olduğu resmin id degeri
             finding - raporlardaki bulgumuz yani asıl raporu olusturan kısımdır.
    '''
    rows = []  # Değerlerimizi tutacak olan satır.

    report_as_csv = open('report_lastfinal.csv', 'a', encoding='utf-8')

    # Değerler satırını oluştur ve csv içerisine yükle.
    rows.append(report_id)
    rows.append(finding)
    csv.writer(report_as_csv).writerow(rows)

    # close openin file .
    report_as_csv.close()
dir = 'C:/Users/umtsr/Desktop/BM495/Veriseti/reports'
with open('reports.csv', 'w', encoding='utf-8') as r:
    columns = ['Id', 'Findings']
    csv.writer(r).writerow(columns)

for file_name in os.listdir(dir):
    if file_name.endswith('.xml'):
        root = xmlRaporuAl(file_name)
        print(file_name + " is loaded.\n")
    else:
        continue
    try:
        my_images_all_id = id_Bul()
        finding = finding_Bul()
        print("Report Id/s:", my_images_all_id,
              "\nFindings:", finding, "\n")
    except:
        print("Exception : Report informations can't get  !!")
    try:
        for id in my_images_all_id:
            satirEkle(id, finding)
            print(id, " is appended successfully.\n\n*********\n")
        if my_images_all_id < 1:
            print("There is NO IMAGE data with related ", file_name)
    except:
        print("Exception : Values can't add to new report file.")

import numpy as np

# Genel dosya işlemleri
from pickle import dump,load
from os import getcwd,listdir,chdir
import csv,string

# NN
from keras.models import Model,load_model
from keras.layers import Input,Dense,Dropout
from keras.layers.merge import add
from keras.utils import to_categorical,plot_model

# CNN
from keras.applications.vgg19 import VGG19,preprocess_input

# RNN
from keras.layers import LSTM,Embedding,Bidirectional


# Preprocessing
from keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


# Fine-tuning
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.optimizers import Adam


# Evaluation
from nltk.translate.bleu_score import corpus_bleu


class Main:
    def run(self):
        # Proje içerisinde kullanılacak dosyaların yolları ve üretilecek dosyaların kayıt yolları.
        # Veri setinin bulunduğu dosya yolu
        images_path = '/content/drive/MyDrive/BM495_141180059/veriseti/NLMCXR_png'
        reports_path = '/content/drive/MyDrive/BM495_141180059/report_lastfinal_sorted.csv'

        # Üretilecek özellik vektörü, sözlük, verisetlerindeki görüntülerin dosya isimlerinin kaydedileceği klasör.
        feature_path = '/content/drive/MyDrive/RADGOR_v1/models/'
        tokenizer_path = '/content/drive/MyDrive/RADGOR_v1/models/'
        train_images_filenames_path = '/content/drive/MyDrive/RADGOR_v1/models/'
        dev_images_filenames_path = '/content/drive/MyDrive/RADGOR_v1/models/'
        test_images_filenames_path = '/content/drive/MyDrive/RADGOR_v1/models/'

        loadOp = loadOperations()
        dataPrepOp = dataPreparationOperations()
        seqOp = sequenceOperations()

        # --- LOAD OPERATIONS --- #

        self.wholeImagesId = loadOp.getAllImageFileNames(
            images_path)  # Tüm resimlerin dosya isimlerini aldim.(Image Id)
        reports_dict_init = loadOp.loadReportsFile(reports_path)  # Tüm raporları aldim. Raporlar = [IMG_ID, Findings]

        ## --- DATA PREPARATION OPERATIONS --- ##
        # TEXT DATA
        reports_dict_current = dataPrepOp.createCurrentReportDict(self.wholeImagesId,
                                                                  reports_dict_init)  # Raporları güncelledim
        findings_init = dataPrepOp.get_findings(reports_dict_current)  # Findingsleri aldım.
        findings_current = dataPrepOp.clean_findings(findings_init)  # Findingslere temizleme işlemleri uyguladım.

        # Sequence başlangıcı ve bitişi için <start>,<end> anahtar kelimelerini ekledim.
        self.reports_final = dataPrepOp.addStartStopKeys(self.wholeImagesId, findings_current)
        findings_final = dataPrepOp.get_findings(self.reports_final)

        # IMAGE DATA
        # Daha önce sorunsuzca çalıştırdım. Koddaki başka hatadan dolayı yeniden çalıştırmadım. Maliyetli.
        # imageFeatures = dataPrepOp.extract_features(images_path) # Tüm görüntülerin feature vektörlerini çıkardım.
        imageFeaturesFile = feature_path + 'wholeFeatures.pkl'  # Feature vektörlerinin dosya ismi
        # loadOp.save_as_pkl(imageFeatures,imageFeaturesFile) # İlgili vektörleri kaydet.
        self.all_images_features = loadOp.load_features(self.wholeImagesId, imageFeaturesFile)

        ### --- SEQUENCE OPERATIONS --- ###

        # Yeni findingsler için Tokenizer yardımıyla sözlük yapısını oluşturdum ve kaydettim

        self.tokenizer = seqOp.createTokenizer(findings_final)  #
        # Daha önce sorunsuzca çalıştırdım. Koddaki başka hatadan dolayı yeniden çalıştırmak istemedim.Maliyetli.
        tokenizerFile = tokenizer_path + 'tokenizer.pkl'
        loadOp.save_as_pkl(self.tokenizer, tokenizerFile)

        # Sequence oluşturmak için gerekli olan Max kelime sayısı, Sözlük ve Sözlük boyutu değerlerini hesapladım.
        self.max_length = seqOp.findMax(findings_final)  # En uzun cümledeki kelime sayısını hesapladım.
        vocab = seqOp.create_vocab(findings_final)  # Sözlük oluşturdum.
        self.n_vocab = len(vocab) + 1  # Sözlüğün boyutunu hesapladım.

        # Define train and dev set size.
        wholeSize = len(findings_final)
        dev_size = int(np.floor(wholeSize * 0.05))
        train_size = wholeSize - dev_size

        # Train set
        train_wholeImagesId = self.wholeImagesId[:train_size]  # Image Id
        train_findings = findings_final[:train_size]  # Findings
        train_reports = dataPrepOp.createNewReports(train_wholeImagesId,
                                                    train_findings)  # Başlangıç ve bitiş kelimeleri ekledim.
        train_features = loadOp.load_features(train_wholeImagesId,
                                              imageFeaturesFile)  # Image id'leri ve bu görüntülerin özellik vektörlerini içeren dict yapısı oluşturdum.

        # Dev set
        self.dev_wholeImagesId = self.wholeImagesId[train_size:]
        dev_findings = findings_final[train_size:]
        self.dev_reports = dataPrepOp.createNewReports(self.dev_wholeImagesId, dev_findings)
        self.dev_features = loadOp.load_features(self.dev_wholeImagesId, imageFeaturesFile)

        self.train_imageFeatures, self.train_textSequences, self.train_nextWord = seqOp.create_sequences(self.tokenizer,
                                                                                                         self.max_length,
                                                                                                         train_reports,
                                                                                                         train_features,
                                                                                                         self.n_vocab)

        self.dev_imageFeatures, self.dev_textSequences, self.dev_nextWord = seqOp.create_sequences(self.tokenizer,
                                                                                                   self.max_length,
                                                                                                   self.dev_reports,
                                                                                                   self.dev_features,
                                                                                                   self.n_vocab)
        # Train ve Dev set için Sequence oluşturdum. Bunu sonucunda bana image feature,sequence ve sonraki kelimeyi verdi. Bunları da Model eğitirken kullanacağım.
        # Şöyle düşünürsek; [imagefeaturel ve text_sequence] = X , sonraki(tahmin edilen) kelime = y.
        print("-Findings:", findings_final[0])
        print("-Max Len:", self.max_length)

    def run4training(self):
        train = TrainingStep()
        train.define_model(self.n_vocab, self.max_length)
        history = train.beginTrain(self.n_vocab, self.max_length, self.train_imageFeatures, self.train_textSequences,
                                   self.train_nextWord, self.dev_imageFeatures, self.dev_textSequences,
                                   self.dev_nextWord)
        return history

    def run4trainingMORE(self):
        train = TrainingStep()
        last_model = '/content/drive/MyDrive/RADGOR_v1/models/model5_h-ep035-loss2.055-val_loss2.310.h5'
        history = train.continueTrain(self.train_imageFeatures, self.train_textSequences, self.train_nextWord,
                                      self.dev_imageFeatures, self.dev_textSequences, self.dev_nextWord, last_model)
        return history

    def sitAndwatch(self):
        eval = EvaluationModel()
        loadOp = loadOperations()
        model = load_model('/content/drive/MyDrive/RADGOR_v1/models/model5_d-ep041-loss2.457-val_loss2.531.h5')
        eval.evaluate_model(model, self.dev_reports, self.dev_features, self.tokenizer, self.max_length)
        eval.evalBLEU()

    def uret(self):
        rg = ReportGeneration()
        my_new_image = '/content/drive/MyDrive/BM495_141180059/veriseti/NLMCXR_png/CXR1000_IM-0003-1001.png'
        photo = rg.extract_features_new_image(my_new_image)
        my_vgg19_model = load_model('/content/drive/MyDrive/RADGOR_v1/models/model2-ep014-loss2.048-val_loss2.307.h5')
        new_finding = rg.generate_desc(my_vgg19_model, self.tokenizer, photo, self.max_length)
        print("* NEW FINDING: ", new_finding)


class loadOperations:
    def getAllImageFileNames(self, filepath):
        '''
        fonksiyon tüm radyolojik görüntülerin dosya isimlerini alır ayırır, kısaca tüm image id leri alıp döndüren fonksiyon
        girdi : görüntülerin olduğu klasör yolu
        çıktı : list
        '''
        filenames = list()
        for filename in listdir(filepath):  # tüm dosyaları dolaş
            if filename.endswith('.png'):  # .png ile bitenleri bul
                filenames.append(filename.split('.')[0])  # mevcut dosyalara ekle.
        print("Images File names got.Files count:", len(filenames))
        return filenames

    def loadReportsFile(self, filepath):
        '''
        Bu fonksiyon tüm raporların içindeki image id ve finding değerlerini döndürür.
        girdi : raporların olduğu dosyanın yolu
        çıktı : findings ve img id leri içeren dict yapısı döndürür.
        '''
        reports_dictionary = dict()
        with open(filepath, 'r') as file:  # csv dosyasını aç
            reader = csv.reader(file, delimiter=',')  # csv dosyasını oku
            for line in reader:
                img_id = line[0]
                finding = line[1]
                reports_dictionary[img_id] = finding
        print("Reports file is loaded. All reports counts:", len(reports_dictionary))
        return reports_dictionary

    def save_as_pkl(self, pkl_file, filename):
        '''
        Bu fonksiyon veri setindeki herhangi bir sözlük yapısını pkl formatında kaydeder.
        Kullanım amacı : Görüntü verilerinden çıkarılan görüntü featurelarının
        ilgili image idlerle eşleştirilmiş liste yapısını ve tokenizer findings yapısının kaydedilmesi.
        '''
        try:
            dump(pkl_file, open(filename, 'wb'))  # kaydet.
            print(f"{filename} is saved")
        except:
            raise Exception(f'{filename} can not be saved..!')

    def load_features(self, wholeImagesId, feature_file):
        '''
        Bu fonksiyon daha önceden çıkarılmış feature'ları yükler.
        input : feature dosyasını ve tüm resimlerin image_idlerini alır.
        output : Tüm image_id değerlerini features dosyası içinde arayarak bulur ve bunu bir dict yapısına kaydeder. Yukarıda oluşturduğumuz yapıyı tekrar oluşturur.
        '''
        features = load(open(feature_file, 'rb'))  # aç
        features_dict = dict()
        for img_id in wholeImagesId:  # id'ye göre featureları bul ve eşleştir.
            features_dict[img_id] = features[img_id]
        print("Feature is loading.. Feature count:", len(features_dict))
        return features_dict

class dataPreparationOperations:

        ### ------ PREPARATION METHODS FOR TEXT DATA ----- ###


    def createCurrentReportDict(self, all_Images, reports_dict_init):
        '''
        bu fonksiyon yeni bir dict yapısı oluşturur. Şöyle ki yukarıdaki incelemeye göre mevcut img idlere göre, olmayanalrın raporlarını atar.
        girdiler : tüm img id'ler ve başlangıçta oluşturulan rapor dict yapısı
        çıktı : mevcut raporların dict yapısı
        '''
        reports_dict_final = dict()
        for img_id, finding in reports_dict_init.items():  # tüm raporları img id ve finding'e göre dolaş
            if img_id in all_Images:  # img_id tüm id'lerin içinde varsa raporda kalsın yoksa sil.
                reports_dict_final[img_id] = finding
        print("New Reports dict elements' count:", len(reports_dict_final))
        return reports_dict_final


    def get_findings(self, reports_dictionary):
        '''
        Sadece raporlardaki findings'leri bir değişkene almak ya da dosyaya kaydetmek için kullanalabilirim.
        girdi : raporların olduğu en güncel dict yapısı
        çıktı : tüm findings olduğu bir liste
        '''
        findings = list()
        for img_id in reports_dictionary.keys():  # tüm raporların findings kısmını bul.
            finding = reports_dictionary[img_id]
            findings.append(finding)
        print("Findings count:", len(findings))
        return findings


    def clean_findings(self, findings_all):
        '''
        Tüm text dosyasını alır. Bizim için findingsler ve onları temizleyip yeni hallerine güncellememi sağlayan fonksiyondur.
        '''
        findings_all = [word.lower() for word in findings_all]  # kücük harf yapar.
        findings_all = [word.translate(str.maketrans('', '', string.punctuation)) for word in
                        findings_all]  # noktalama işaretlerini siler. String.puncuation bize tüm noktalama işaretlerinin olduğu listeyi verir.
        print("Clean findings count:", len(findings_all))
        return findings_all

        ### --- PREPARATION METHODS FOR IMAGE DATA --- ###


    def extract_features(self, images_path):
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


    def addStartStopKeys(self, wholeImagesId, findings_init):
        '''
        Bu fonksiyonla sequence oluşturmadan önce findings kısmına eklemeler yapıldı. Bu sayede sequence'in başlangıc ve bitişi rahatça sağlanabilir.
        Input : tüm temizlenmiş(kücükharf/noktalama işaretsiz) ve tüm image id leri alır.
        çıktı : başlangıç ve bitiş kelimesi verilmiş yeni sequenceleri oluşturur.
        '''
        reports_final = dict()
        for i in range(len(findings_init)):
            finding = '<start> ' + findings_init[
                i] + ' <end>'  # Evaluation ya da prediction yaparken cümle başlangıcı ve bitişinde kullanılır.
            img_id = wholeImagesId[i]
            reports_final[img_id] = finding
        return reports_final


    def createNewReports(self, wholeImagesId, findings):
        reports_final = dict()
        for i in range(len(findings)):
            img_id = wholeImagesId[i]
            finding = findings[i]
            reports_final[img_id] = finding
        return reports_final

class sequenceOperations:
  def create_vocab(self,findings_all):
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
  def createTokenizer(self,findings_all_clean):
    '''
    fit_on_text sözlük oluşturmak için kullanılan alternatif bir yol tokenizer sınıfından faydalanılarak kolayca halledilebiliyor.
    '''
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(
      findings_all_clean)  # text'i kendisinin kullanabileceği formata çeviriyor. İçeriği yazdırmayı denedim ama göremedim.
    return tokenizer

  def findMax(self,findings_all_clean):
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


  def create_sequences(self,tokenizer, max_length, findings_map, features_map, vocab_size):
    '''
    Bu fonksiyon sequence oluşturur. Tokenizer bizim yazıyı(findings) işlenebilir hale getirdiğimiz nesne.
    Input: tokenizer : findingslerden oluşan token
          max_length: en uzun sequence eleman sayısı
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
      sequence = tokenizer.texts_to_sequences([finding])[0] #finding'i kelimelere ayırıp, sequence haline getiriyor..Diğer deyişle her kelime uzun bir listeye alınıyor.
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


class TrainingStep:
    def define_model(self, n_vocab, max_length):
        # Encoder
        LOSS = 'categorical_crossentropy'
        OPT = Adam(learning_rate=1e-5)  # default = 1e-3
        METRIC = [
            'accuracy']  # Accuracy verdiğinde keras otomatik olarak veritipni bulup parmetre ayarlıyor. Burada categorical accuracy kullanıyor.
        # feature extractor model
        extracted_features = Input(shape=(4096,))  # imageFeatures.shape[1]
        # Burası decoder katmanında merge ederken 4096 ile 256 eleman birleştirlemediği için var. Dropout ile de ihtimalleri daha iyileştiriyorum. Tekrar bir eğitim yok vektörizasyon var.
        encImage_dropout = Dropout(0.5)(extracted_features)
        encImage_dense = Dense(128, activation='relu')(encImage_dropout)

        # sequence model
        finding_sequences = Input(shape=(max_length,))  # sequenceFeatures.shape[1]
        encSeq_embedding = Embedding(n_vocab, 128, mask_zero=True)(
            finding_sequences)  # mask_zero=True , Padding uyguladığımızı ve 0 ları görmezden gelmemizi sağlayan Mask katmanı ekler.
        encSeq_dropout1 = Dropout(0.5)(encSeq_embedding)
        encSeq_LSTM = LSTM(128)(encSeq_dropout1)
        encSeq_dropout2 = Dropout(0.5)(encSeq_LSTM)

        # decoder model
        mergeLayers = add([encImage_dense,
                           encSeq_dropout2])  # Toplama işlemiyle merge ederek daha iyi sonuçlar alındığından "add" kullanılmıştır.
        decoder = Dense(128, activation='relu')(mergeLayers)
        outputs = Dense(n_vocab, activation='softmax')(decoder)
        model = Model(inputs=[extracted_features, finding_sequences], outputs=outputs)
        model.compile(loss=LOSS, optimizer=OPT, metrics=METRIC)
        # summarize model
        print(model.summary())
        plot_model(model, to_file='model.png', show_shapes=True)
        return model

    def beginTrain(self, n_vocab, max_length, train_Ximages, train_Xsequences, train_y, dev_Ximages, dev_Xsequences,
                   dev_y):
        # define the model
        BATCH_SIZE = 64
        EPOCH = 500
        model = self.define_model(n_vocab, max_length)
        # define checkpoint callback
        current_dir = getcwd()
        vgg19_model_path = '/content/drive/MyDrive/RADGOR_v1/models/'
        filepath = vgg19_model_path + 'model5-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'

        # Fine-tuning.
        cb_checkpoint = ModelCheckpoint(filepath,
                                        monitor='val_loss',
                                        verbose=1,
                                        save_best_only=True,
                                        mode='min')
        cb_early_stopping = EarlyStopping(patience=3, restore_best_weights=True)

        # fit model
        print(f"Model fit başlatıldı.\nBATCH SIZE:{BATCH_SIZE}\nEPOCH:{EPOCH}\n")
        history = model.fit([train_Ximages, train_Xsequences],
                            train_y,
                            epochs=EPOCH,
                            batch_size=BATCH_SIZE,
                            validation_data=([dev_Ximages, dev_Xsequences], dev_y),
                            callbacks=[cb_checkpoint, cb_early_stopping])
        return history

    def continueTrain(self, train_Ximages, train_Xsequences, train_y, dev_Ximages, dev_Xsequences, dev_y, lastmodel):
        # define the model
        BATCH_SIZE = 64
        EPOCH = 500
        model = load_model(lastmodel)
        # define checkpoint callback
        current_dir = getcwd()
        vgg19_model_path = '/content/drive/MyDrive/RADGOR_v1/models/'
        filepath = vgg19_model_path + 'model5_j-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'

        # Fine-tuning.
        cb_checkpoint = ModelCheckpoint(filepath,
                                        monitor='val_loss',
                                        verbose=1,
                                        save_best_only=True,
                                        mode='min')
        cb_early_stopping = EarlyStopping(patience=3, restore_best_weights=True)

        # fit model
        print(model.summary())
        print(f"Model fit TEKRAR başlatıldı.\nBATCH SIZE:{BATCH_SIZE}\nEPOCH:{EPOCH}\n")
        history = model.fit([train_Ximages, train_Xsequences],
                            train_y,
                            epochs=EPOCH,
                            batch_size=BATCH_SIZE,
                            validation_data=([dev_Ximages, dev_Xsequences], dev_y),
                            callbacks=[cb_checkpoint, cb_early_stopping])
        return history
class ReportGeneration:
  def word2id(self,integer,tokenizer):
    for word, index in tokenizer.word_index.items():
      if index == integer:
        return word
    return None

  # generate a description for an image
  def generate_desc(self,model, tokenizer, image, max_length):
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
        yhat = model.predict([image,sequence]) # verbose Ne yapıyor ?
        print(f"{i}. prediction complete.")
      except:
        raise Exception('Prediction hatası!')
      # convert probability to integer
      yhat = np.argmax(yhat)
      # map integer to word
      word = self.word2id(yhat, tokenizer)
      # stop if we cannot map the word
      if word is None:
        break
      # append as input for generating the next word
      in_text += ' ' + word
      # stop if we predict the end of the sequence
      if word == '<end>':
        break
    return in_text


  def extract_features_new_image(self,filename):
      # load the model
      model = VGG19()
      # re-structure the model
      model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
      # load the photo
      image = load_img(filename, target_size=(224, 224))
      # convert the image pixels to a numpy array
      image = img_to_array(image)
      # reshape data for the model
      w,h,ch = image.shape
      image = image.reshape((1,w,h,ch))
      # prepare the image for the VGG model
      image = preprocess_input(image)
      # get features
      feature = model.predict(image, verbose=0)
      return feature
class EvaluationModel:
  def evaluate_model(self,model, reports, features, tokenizer, max_length):
      RG = ReportGeneration()
      print("Evaluate model çalışıyor..")
      self.y_all, self.yhat_all = list(), list()
      # step over the whole set
      i = 1 #step count
      for img_id, finding in reports.items():
        yhat = RG.generate_desc(model, tokenizer, features[img_id], max_length)
        print(f"{i}. step : generating 'findings' for: {img_id}")
        i +=1
        references = [finding.split()]
        predicted_references = yhat.split()
        print(f"FINDING: {finding}\nREFERENCES:{references}")
        print(f"PREDICTED REFERENCES: {predicted_references}")
        self.y_all.append(references)
        self.yhat_all.append(predicted_references)

  def evalBLEU(self):
    y_all = self.y_all
    yhat_all = self.yhat_all
    print('BLEU-1: %f' % corpus_bleu(y_all, yhat_all, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(y_all, yhat_all, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(y_all, yhat_all, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(y_all, yhat_all, weights=(0.25, 0.25, 0.25, 0.25)))


letsgo = Main()
letsgo.run()
history = letsgo.run4training()
letsgo.sitAndwatch()

import matplotlib.pyplot as plt

plt.subplot(212)
plt.plot(history.history['loss'],label='Train')
plt.plot(history.history['val_loss'],label='Dev')
plt.legend()
plt.title("Loss")
plt.subplot(211)
plt.plot(history.history['accuracy'],label='Train')
plt.plot(history.history['val_accuracy'],label='Dev')
plt.legend()
plt.title("Accuracy")
plt.show()