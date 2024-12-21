#!/usr/bin/env python
# coding: utf-8

import requests
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
from io import BytesIO

url = './data/Clams/10711395_a16c4c2901_o.jpg'

interpreter = tflite.Interpreter(model_path='sea-creature-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = [
    'Clams',
    'Corals',
    'Crabs',
    'Dolphin',
    'Eel',
    'Fish',
    'Jelly Fish',
    'Lobster',
    'Nudibranchs',
    'Octopus',
    'Otter',
    'Penguin',
    'Puffers',
    'Seahorse',
    'Sea Rays',
    'Sea Urchins',
    'Seal',
    'Sharks',
    'Shrimp',
    'Squid',
    'Starfish',
    'Turtle_Tortoise',
    'Whale'
]

def preprocess_input(x):
  x /= 127.5
  x -= 1.
  return x

def predict(url):
  if url.startswith('http://') or url.startswith('https://'):
    
    response = requests.get(url)
    response.raise_for_status()
    with Image.open(BytesIO(response.content)) as img:
      img = img.resize((299, 299), Image.NEAREST)
  else:
    with Image.open(url) as img:
      img = img.resize((299, 299), Image.NEAREST)

  x = np.array(img, dtype='float32')
  X = np.array([x])
  X = preprocess_input(X)

  interpreter.set_tensor(input_index, X)
  interpreter.invoke()
  preds = interpreter.get_tensor(output_index)

  float_predictions = preds[0].tolist()

  return dict(zip(classes, float_predictions))

def lambda_handler(event, context):
  url = event['url']
  result = predict(url)
  return result
