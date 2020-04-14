import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD,Adam
import os.path
import json
from keras.models import model_from_json
import matplotlib.pyplot as plt


##################################################################################

# For reproducibility - splitting train and test sets
seed = 12
np.random.seed(seed)

dataset=pd.read_csv('test.csv')


dataset=pd.read_csv('train.csv')

dataset2=dataset

col=dataset.columns

dataset2=dataset.drop(['PassengerId','Name','Cabin','Ticket'], axis=1)

dataset3=dataset2.dropna(axis=0, how='any')

col=dataset2.columns

dataset2=dataset2.reset_index(drop=True)

dataset2

for i in range(len(dataset2['Sex'])):
    print(i)

    if dataset2['Sex'][i] =='male':
        dataset2['Sex'][i]=0
    elif dataset2['Sex'][i] =='female':
        dataset2['Sex'][i]=1
    
    if dataset2['Embarked'][i] =='S':
        dataset2['Embarked'][i]=0
    elif dataset2['Embarked'][i] =='C':
        dataset2['Embarked'][i]=1
    elif dataset2['Embarked'][i] =='Q':
        dataset2['Embarked'][i]=2


y_dataset=dataset2['Survived']
x_dataset=dataset2.drop(['Survived'], axis=1)



y_dataset=dataset['PassengerId']
x_dataset=dataset2

col=x_dataset.columns


for i in col:
    avg=x_dataset[str(i)].mean()
    sd=x_dataset[str(i)].std()
    x_dataset[str(i)]=x_dataset[str(i)].apply(lambda X:(X-avg)/(sd))
    print(avg)
    print(sd)
    print(i)




    
print("Normalized Data\n", x_dataset[:5], "\n")

# covert to array for processing
x_dataset=x_dataset.values

#One hot encoding
#_complete= pd.get_dummies(y_complete).values
y_dataset=y_dataset.values
y_complete2= pd.get_dummies(y_dataset).values


# Creating a Train and a Test Dataset
X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_complete2, test_size=0.3, random_state=seed)


# Define Neural Network model layers
model = Sequential()
model.add(Dense(10, input_dim=7, activation='relu'))
#model.add(Dense(10, input_dim=11, activation='softmax'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile model
model.compile(Adam(lr=0.001),'categorical_crossentropy',metrics=['accuracy'])




if os.path.isfile('@dead_or_alive.h5'):

    # Model reconstruction from JSON file
    json_file = open('dead_or_alive.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    # Load weights into the new model
    model.load_weights('dead_or_alive.h5')
    print("Model weights loaded from saved model data.")

    model.compile(Adam(lr=0.001),'categorical_crossentropy',metrics=['accuracy'])
else:
    print("Model weights data not found. Model will be fit on training set now.")

    # Fit model on training data - try to replicate the normal input
    history=model.fit(X_train,y_train,epochs=1000,batch_size=100,verbose=1,validation_data=(X_test,y_test))
    
 
         # Save parameters to JSON file
    model_json = model.to_json()
    with open("dead_or_alive.json", "w") as json_file:
        json_file.write(model_json)

    # Save model weights to file
    model.save_weights('dead_or_alive.h5')

model.summary()




# Model predictions for test set
y_pred = model.predict(x_dataset)
y_test_class = np.argmax(y_complete2,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)

print(y_test_class,y_pred_class)
#print(y_pred_class)


# Evaluate model on test data
score = model.evaluate(x_dataset,y_complete2, batch_size=128,verbose=1)
 
# Compute stats on the test set and Output all results
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test_class,y_pred_class))
print(confusion_matrix(y_test_class,y_pred_class))





plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['train', 'test'], loc='upper left')

###########################################################################

y_dataset
output_dataset=pd.DataFrame(y_pred_class)


frames=[y_dataset,output_dataset]

dataset_out=pd.concat(frames, axis=1)

dataset_out.to_csv('test_preds.csv')

dataset2.to_csv('training_dataset_conditioned.csv')


