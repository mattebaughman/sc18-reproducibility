from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import History
import lstm, time #helper libraries
import time
import numpy as np
#import random
#Step 1 Load Data
history = History()
p=time.time()
X_train, y_train, X_test, y_test = lstm.load_data('spot1.csv', 15, True)
#Step 2 Build Model
weights = []

for p in range(4):
	model = Sequential()
	model.add(LSTM(input_dim=1,output_dim=2,return_sequences=False))
	#model.add(Dense(output_dim=32))
	#model.add(Dense(output_dim=32))
	#model.add(LSTM(output_dim=2,return_sequences=False))
	model.add(Dense(output_dim=1))
	model.add(Activation('relu'))



	model.compile(loss='mse', optimizer='sgd')
	hist  = model.fit(
          X_train,
          y_train,
          batch_size=80096,
          nb_epoch=4,
          validation_split=.01,
	  callbacks=[history]
	)
	weights.append(model.get_weights())
	if p ==3:
		old_weights = model.get_weights()
#print(weights)
#print(time.time()-p)
#Step 4 - Plot the predictions!
#predictions = lstm.predict_sequences_multiple(model, X_test, 50, 50)
#lstm.plot_results_multiple(predictions, y_test, 50)
#print(hist.history['val_loss'])
#print(model.get_weights())
#new_weights = np.mean(weights[0][1],weights[1][1])
top_two = []
'''
for l in range(2):
	top = [100000000,1000000000]
	for k in range(len(weights)):
		print(weights[k][0][0])
		if weights[k][0][0]<top[0]:
			top = [weights[k][0][0],k]
	top_two.append(weights[top[1]])
	
'''	
#print(top_two)		
new_weights = []

for weights_list_tuple in zip(*weights):
    new_weights.append(numpy.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple))

print(new_weights)
print(old_weights)
for k in (new_weights):
	print(k)

new_weights = [np.array([[-0.43,  -0.30,  0.03,  0.15,  0.75, 0.29, 0.12 , -0.68]]),np.array([[ 0.57,  0.29,  0.13,  0.36,  0.10, -0.01, 0.05, -0.25],[ 0.10, -0.20, -0.20,  0.08,  0.075,-0.52, -0.10, -0.21 ]]), np.array([ -1.8e-06,  -1.6e-06,   .99, .99, 1.4e-04, 7.4e-05, 8.3e-06, -9.1e-05]), np.array([[ 1.14], [ 0.70]]), np.array([  2.09e-04])]
model.set_weights(new_weights)

model.fit(
          X_train,
          y_train,
          batch_size=80096,
          nb_epoch=2,
          validation_split=.01)
