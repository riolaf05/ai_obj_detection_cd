## Istructions
TODO

## Algorithm

### Preprocess 

1. Create a set of document with the format: `[["vocabulary words"], intent]`

2. Create a list of intents (classes)

3. Create a list of vocabulary words (words)

4. Create a binary class matrix where for each intent I get a list with 1 in the intent corresponding position, e.g.

```console
intent                  class matrix
"goodbye"         -->   1 0 0 
"hello"           -->   0 1 0
"pharmacy_search" -->   0 0 1 
```

4. For each vocabulary word, create a list (bag) with index=1 if the word correspond to the intent 

### Training

The architecture of our model will be a neural network consisting of 3 dense layers. The first layer has 128 neurons, the second one has 64 and the last layer will have the same neurons as the number of classes. The dropout layers are introduced to reduce overfitting of the model. We have used the SGD optimizer and fit the data to start the training of the model. After the training of 200 epochs is completed, we then save the trained model using the Keras model.save(“chatbot_model.h5”) function.

### inference
TODO

## Next steps
* Add MLFlow for packaging
* Add entity recognition

## References
[Build your first chatbot with Python](https://dzone.com/articles/python-chatbot-project-build-your-first-python-pro)

