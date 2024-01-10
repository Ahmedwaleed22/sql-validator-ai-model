import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.mixed_precision import set_global_policy
import numpy as np

"""
Start Should Be Commented Part After Training

and replaced with

# # Load the trained model
# model_path = 'path/to/your/siamese_model.h5'  # Update with the correct path
# siamese_model = tf.keras.models.load_model(model_path)
"""
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
# Define Siamese model
def build_siamese_model():
    input_1 = Input(shape=(512,), dtype='int32')
    input_2 = Input(shape=(512,), dtype='int32')

    # Shared BERT model
    bert_layer = bert_model.layers[0]
    
    # Process each input query through the same BERT model
    bert_output_1 = bert_layer(input_1)[1]  # We take the pooled output
    bert_output_2 = bert_layer(input_2)[1]

    # Neural network for processing BERT outputs
    siamese_layer = Dense(64, activation='relu')
    processed_1 = siamese_layer(bert_output_1)
    processed_2 = siamese_layer(bert_output_2)

    # Lambda layer to calculate the distance
    l1_distance_layer = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
    l1_distance = l1_distance_layer([processed_1, processed_2])

    # Dense layer to make a final prediction
    prediction = Dense(1, activation='sigmoid')(l1_distance)

    # Create the Siamese Network model
    siamese_network = Model(inputs=[input_1, input_2], outputs=prediction)

    return siamese_network

# Build and compile the model
siamese_model = build_siamese_model()
siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

"""
End Should Be Commented Part
"""

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def bert_preprocess_query(query):
    encoded = tokenizer.encode_plus(
        query,
        add_special_tokens=True,
        max_length=512,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='tf',
    )
    return encoded['input_ids'][0]

def combine_question_with_query(question, query):
    combined_text = question + " " + query  # or any other combination logic you prefer
    return bert_preprocess_query(combined_text)

def compare_queries(question, query1, query2, model):
    combined_query1 = combine_question_with_query(question, query1)
    combined_query2 = combine_question_with_query(question, query2)

    # The model now receives two inputs as expected
    prediction = model.predict([combined_query1[tf.newaxis, :], combined_query2[tf.newaxis, :]])
    return prediction[0][0]

