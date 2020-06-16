import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import os
import telepot

BASE_DIR='/src/models'
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

from tensorflow.keras.models import load_model
model = load_model(os.path.join(BASE_DIR, 'chatbot_model.h5'))
import json
import random
intents = json.loads(open(os.path.join(BASE_DIR, 'intents.json')).read())
words = pickle.load(open(os.path.join(BASE_DIR, 'words.pkl','rb')))
classes = pickle.load(open(os.path.join(BASE_DIR, 'classes.pkl','rb')))

def clean_up_sentence(sentence):
    # tokenize the pattern - splitting words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stemming every word - reducing to base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for words that exist in sentence
def bag_of_words(sentence, words, show_details=True):
    # tokenizing patterns
    sentence_words = clean_up_sentence(sentence)
    # bag of words - vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))

def predict_class(sentence):
    # filter below  threshold predictions
    p = bag_of_words(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def predict_response(msg):
  ints = predict_class(msg)
  res = getResponse(ints, intents)
  return res

def on_chat_message(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type == 'text':
        #name = msg["first_name"]
        txt = msg['text']
        res=predict_response(txt)
        bot.sendMessage(chat_id, res)

bot = telepot.Bot(TOKEN)
bot.message_loop(on_chat_message)

print('Listening ...')

import time
while 1:
    time.sleep(10)