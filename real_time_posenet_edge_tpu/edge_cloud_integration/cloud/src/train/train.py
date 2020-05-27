import os
import pandas
import argparse
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import tensorflow as tf
from google.cloud import storage
from sklearn.model_selection import train_test_split
from tensorflow.python.lib.io import file_io

def read_from_gcs(bucket, filename, path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(filename)
    blob.download_to_filename(os.path.join(path, filename))
    print("Data downloaded to ", os.path.join(path, filename))

def write_to_gcs(bucket, filename, path):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(filename)
    blob.upload_from_filename(os.path.join(path, filename))
    print(path+filename, " Model written to ", bucket, " bucket.")

# define baseline model
def baseline_model(inputdim, outputdim):
	# create model
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(8, input_dim=inputdim, activation='relu'))
	model.add(tf.keras.layers.Dense(outputdim, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def categorical(y_train):# encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    # convert integers to dummy variables (i.e. one hot encoded)
    train_y_categorical = np_utils.to_categorical(encoded_Y)
    return train_y_categorical

def main():
    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='Train arguments')
    parser.add_argument('--input-bucket', type=str, help='Input bucket.') 
    parser.add_argument('--output-bucket', type=str, help='output bucket') 
    args = parser.parse_args()

    read_from_gcs(args.input_bucket, "final_df.csv", "/")

    dataframe = pandas.read_csv("/final_df.csv", index_col=[0])

    #Trainind data
    train_X = dataframe.drop(columns=['pose'])
    train_X=train_X.to_numpy()

    #Labels
    Y = dataframe['pose']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

    # encode class values as integers
    train_y_categorical=categorical(y_train)
    
    model=baseline_model(train_X.shape[1], train_y_categorical.shape[1])
    model.fit(train_X, train_y_categorical, batch_size=5, validation_split=0.2, epochs=200)
    model.save('/activity_classification.h5')
    write_to_gcs(args.output_bucket, 'activity_classification.h5', '/')

    #Get accuracy/confusion matrix
    y_pred = model.predict_classes(X_test)
    loss, accuracy = model.evaluate(X_train, train_y_categorical, verbose=0)
    y_actu = pd.Series(categorical(y_test)[0], name='target')
    y_pred = pd.Series(y_pred, name='predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    df_confusion.to_csv('/conf_matrix.csv')
    metrics = {
        'metrics': [{
        'name': 'accuracy-score', # The name of the metric. Visualized as the column name in the runs table.
        'numberValue':  accuracy, # The value of the metric. Must be a numeric value.
        'format': "PERCENTAGE",   # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
        }]
    }
    with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)
    

if __name__ == "__main__":
    main()
        
