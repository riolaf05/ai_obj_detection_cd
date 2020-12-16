import os
import telepot
import argparse
import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama 
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

def chat(model, tokenizer, encoder, data, imp):
    # load trained model
    model = keras.models.load_model(args.model)

    # load tokenizer object
    with open(args.tokenizer, 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open(args.encoder, 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                            truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])

def on_chat_message(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type == 'text':
        #name = msg["first_name"]
        txt = msg['text']
        res=chat(txt)
        bot.sendMessage(chat_id, res)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="chatbot trained model", default="saved_model")
    parser.add_argument("tokenizer", help="tokenizer", default="utils/tokenizer.pickle")
    parser.add_argument("encoder", help="encoder", default="utils/label_encoder.pickle")
    parser.add_argument("intents", help="intents file", default="intents.json")
    args = parser.parse_args()

    with open(args.intents) as file:
        data = json.load(file)

        bot = telepot.Bot(TOKEN)
        bot.message_loop(on_chat_message)

        print('Listening ...')

        import time
        while 1:
            time.sleep(10)

if __name__ == "__main__":
    main()