from tensorflow.contrib import lite

converter = lite.TFLiteConverter.from_keras_model_file( r'models\\1602079553.h5')
tfmodel = converter.convert()

open(r"models\\1602079553.tflite" , "wb").write(tfmodel)