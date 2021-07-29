from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input,Dense,LSTM,Embedding

#Fine-tuning
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint


# define model
# define the captioning model
# Bir veya iki adet daha LSTM ekle.
def define_model(vocab_size, max_length):
    #Encoder
	# feature extractor model
	inputs1 = Input(shape=(4096,)) #X1train.shape[1]
    # Burası decoder katmanında merge ederken 4096 ile 256 eleman birleştirlemediği için var. Dropout ile de ihtimalleri daha iyileştiriyorum. Tekrar bir eğitim yok vektörizasyon var.
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,)) # X2train.shape[1]
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2) # mask zero ?? Modeli düzenlerken parametreleri de incele. # Embedding katmanı tam olarak ne yapıyor ? vocab sayısı,boyutu,input uzunlugu.
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	# decoder model
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# bağlayalım [image, seq] [word] # İki katmanı modelde merge ederken add,concatenate gibi yöntemlerde varmış bunlara da bak.
    # Ayrıca Attention mechanism kısmına da bak. Projede nerede kullanabilirsin katkısı olur mu incele/uygula.
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	print(model.summary())
	plot_model(model, to_file='model.png', show_shapes=True)
	return model

def beginTrain(n_vocab,max_length,X1train,X2train,ytrain,X1dev,X2dev,ydev):

	# define the model
	BATCH_SIZE = 128
	EPOCH = 5
	model = define_model(n_vocab, max_length)
	# define checkpoint callback
	vgg19_model_path = '/models/'
	filepath = vgg19_model_path + 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	# fit model
	try:
	  print("Model fit çalıştırılıyor...")
	  model.fit([X1train, X2train], ytrain, epochs=EPOCH, batch_size=BATCH_SIZE,validation_data=([X1dev, X2dev], ydev),callbacks=[checkpoint])
	except Exception as e:
	  raise Exception("Exception ismi : ",e)

