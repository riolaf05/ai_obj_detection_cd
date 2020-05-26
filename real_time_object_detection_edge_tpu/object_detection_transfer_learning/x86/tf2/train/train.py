import tensorflow as tf
assert tf.__version__.startswith('2')

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def train(base_dir, image_size, batch_size, epochs):
    #creo o generatori
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,  
    validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size, 
        subset='training')

    val_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size, 
        subset='validation')

    #creo il file di label
    labels = '\n'.join(sorted(train_generator.class_indices.keys()))

    with open('/save/labels.txt', 'w') as f:
        f.write(labels)

    IMG_SHAPE = (image_size, image_size, 3)

    # Creo il modello base dal pre-trained model
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False, 
                                                weights='imagenet')
    #freezzo il modello
    base_model.trainable = False

    #creazione modello
    model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

    print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

    history = model.fit(train_generator, 
                        steps_per_epoch=len(train_generator), 
                        epochs=epochs, 
                        validation_data=val_generator, 
                        validation_steps=len(val_generator))

    #fine tuning (aumenta le prestazioni)

    base_model.trainable = True

    fine_tune_at = 100

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    model.compile(loss='categorical_crossentropy',
                optimizer = tf.keras.optimizers.Adam(1e-5),
                metrics=['accuracy'])

    history_fine = model.fit(train_generator, 
                         steps_per_epoch=len(train_generator), 
                         epochs=5, 
                         validation_data=val_generator, 
                         validation_steps=len(val_generator))

    acc = history_fine.history['accuracy']
    val_acc = history_fine.history['val_accuracy']

    loss = history_fine.history['loss']
    val_loss = history_fine.history['val_loss']

    #salvataggio metriche per Kubeflow
    metrics = {
        'metrics': [{
        'name': 'accuracy-score', # The name of the metric. Visualized as the column name in the runs table.
        'numberValue':  acc, # The value of the metric. Must be a numeric value.
        'format': "PERCENTAGE",   # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
        }]
    }
    with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)

    #salvataggio modello
    saved_model_dir = '/save'
    tf.saved_model.save(model, saved_model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, help='cartella con i dati', required=True)
    parser.add_argument('--image-size', type=int, help='dimensione di input di MobileNetV2', required=True, default=224)
    parser.add_argument('--batch-size', type=int, help='dimensione batch size dei generatori', required=True, default=16)
    parser.add_argument('--epochs', type=int, default=10, help='numero di epoche')

    args = parser.parse_args()

    base_dir=os.path.join('/data', args.image_dir)

    train(base_dir, args.image_size, args.batch_size, args.epochs)





