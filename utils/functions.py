import re
import emoji

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import sigmoid

from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, AdamW
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification,DataCollatorWithPadding
from datasets import load_dataset, concatenate_datasets, DatasetDict


# Limpieza de tweet
def preprocess_tweet(tweet):
    """
    Limpia un tweet eliminando menciones, hashtags, emojis y caracteres especiales.
    
    Args:
        tweet (str): El texto del tweet a limpiar.
    
    Returns:
        str: El tweet limpio.
    """
    # Elimina los espacios en blanco al inicio y al final
    tweet = tweet.strip()
    # Elimina las menciones
    tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)
    # Mantiene las palabras de los hashtags pero elimina el carácter hashtag
    tweet = re.sub(r'#(\w+)', r'\1', tweet)
    # Convierte los emojis a texto
    tweet = emoji.demojize(tweet)
    # Elimina caracteres especiales, excepto los dos puntos
    tweet = re.sub(r'[^\w\sáéíóúÁÉÍÓÚÑñ:]', '', tweet)
    # Convierte el tweet a minúsculas
    tweet = tweet.lower()
    # Elimina los espacios en blanco al inicio y al final
    tweet = tweet.strip()
    return tweet


# Ajuste del tokenizador
def process(df):
    """
    Procesa los datos de un DataFrame utilizando un tokenizador.
    
    Args:
        df (pandas.DataFrame): El DataFrame que contiene los datos a procesar.
    
    Returns:
        dict: Un diccionario con los datos tokenizados.
    """
    tokenized_inputs = tokenizer(
       df["Tweet"], padding="max_length", truncation=True
    )
    return tokenized_inputs


# Crea un DataLoader
def create_data_loader(df, tokenizer, max_len, batch_size):
    """
    Crea un DataLoader a partir de un DataFrame para el entrenamiento de un modelo.
    
    Args:
        df (pandas.DataFrame): El DataFrame que contiene los datos a cargar en el DataLoader.
        tokenizer: El tokenizador utilizado para convertir los textos en secuencias de tokens.
        max_len (int): La longitud máxima de las secuencias de tokens.
        batch_size (int): El tamaño del lote (batch) de datos.
    
    Returns:
        torch.utils.data.DataLoader: El DataLoader creado.
    """
    ds = EmotionDataset(
        tweets=df.Tweet.to_numpy(),
        labels=df[emotions].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True
    )


# Evalua el modelo
def evaluate_model(model, data_loader):
    """
    Evalúa un modelo utilizando un DataLoader y devuelve la pérdida promedio.
    
    Args:
        model: El modelo a evaluar.
        data_loader (torch.utils.data.DataLoader): El DataLoader que contiene los datos de evaluación.
    
    Returns:
        float: La pérdida promedio obtenida durante la evaluación.
    """
    model = model.eval()

    final_loss = 0
    counter = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            final_loss += loss.item()
            counter += 1

    return final_loss / counter


# obtener predicciones
def get_predictions(model, data_loader):
    """
    Genera predicciones utilizando un modelo y un DataLoader.
    
    Args:
        model: El modelo utilizado para realizar las predicciones.
        data_loader (torch.utils.data.DataLoader): El DataLoader que contiene los datos de entrada.
    
    Returns:
        tuple: Una tupla que contiene dos arrays numpy, uno con las predicciones y otro con los valores reales.
    """
    model = model.eval()
    predictions = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            preds = torch.sigmoid(outputs.logits)
            preds = preds.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

            predictions.extend(preds)
            real_values.extend(labels)

    predictions = np.array(predictions)
    real_values = np.array(real_values)
    return predictions, real_values