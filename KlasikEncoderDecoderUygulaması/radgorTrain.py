import loadOperations as LO
import prepareData as DP
import radgorModel as RM
import sequenceOperations as SO

images_path = 'C:/Users/umtsr/Desktop/BM495/Veriseti/images/'
report_path = 'C:/Users/umtsr/Desktop/BM495/Veriseti/reports.csv'
feature_path = 'models/'
tokenizer_findings_path = '/models/'

images_id_all = LO.get_images_filenames(images_path) # tüm resimlerin id'lerini bir listeye aldım.
reports_dictionary = LO.load_report(report_path) # eğitim için rapor dosyasını yükledim
reports_dictionary_new = DP.clean_report(images_id_all,reports_dictionary) # rapor içerisinde olup, resimelr klasöründe olmayan resimleri dictionary'den attım.Tekrar eşleştiren bir dict yapısı oluşturdum.
findings_all = DP.get_findings(reports_dictionary_new) # sadece findingsleri alıp liste şeklinde döndürdüm.
findings_all_clean = DP.clean_findings(findings_all) # sözlük oluşturmak için gerekli temizleme işlemleri yaptım.
findings_map = LO.load_findings_all_clean_dict(findings_all_clean ,images_id_all)


features = DP.extract_features(images_path)
features_file_full_path = feature_path + 'features_vgg19.pkl'
DP.save_as_pkl(features, features_file_full_path)


# Alt kısımdaki kodları, feature'lar nasıl yüklenir ve içeriği nasıldır incelemek için uyguladım.
feature_file = features_file_full_path
features = LO.load_features(feature_file ,images_id_all)


train_findings = findings_all_clean[:6000]
train_img_id = images_id_all[:6000]
train_findings_map = LO.load_findings_all_clean_dict(train_findings,train_img_id)
train_features = LO.load_features(feature_file,train_img_id)

dev_findings = findings_all_clean[6000:]
dev_img_id = images_id_all[6000:]
dev_findings_map = LO.load_findings_all_clean_dict(dev_findings,dev_img_id)
dev_features = LO.load_features(feature_file,dev_img_id)

tokenizer_findings = SO.createTokenizer(findings_all_clean)
tokenizer_findings_full_path = tokenizer_findings_path + 'tokenizer_findings.pkl'
DP.save_as_pkl(tokenizer_findings,tokenizer_findings_full_path)


max_length = SO.findMax(findings_all_clean)
vocab = SO.create_vocab(findings_all_clean)
n_vocab = len(vocab) + 1

X1train, X2train, ytrain = SO.create_sequences(tokenizer_findings, max_length, train_findings_map, train_features,
                                            n_vocab)
X1dev, X2dev, ydev = SO.create_sequences(tokenizer_findings, max_length, dev_findings_map, dev_features,
                                         n_vocab)

# Modeli oluşturup eğitime başlayalım. Ağırlıklar kaydedilir.
RM.beginTrain(n_vocab,max_length,X1train,X2train,ytrain,X1dev,X2dev,ydev)

# Evaluate my model
