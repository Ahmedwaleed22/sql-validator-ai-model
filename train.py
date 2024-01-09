import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.mixed_precision import set_global_policy
import numpy as np

set_global_policy('mixed_float16')

# Initialize the tokenizer and the BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

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

# Load the dataset
df = pd.read_csv('dataset.csv', delimiter='|', quotechar='"')

# Correct query
correct_query = "Select ID, Name, salary, ROUND(salary*1.15,0) AS New_Salary From employee;"
preprocessed_correct_query = bert_preprocess_query(correct_query)

# Filter out non-executable queries and prepare data pairs
paired_data = []
labels = []
for _, row in df.iterrows():
    user_query = row['UserQuery']
    preprocessed_user_query = bert_preprocess_query(user_query)

    paired_data.append((preprocessed_user_query, preprocessed_correct_query))
    labels.append(0)  # assuming all user queries are incorrect (label them as 0)

# Convert to tensors
query1_tensors = tf.convert_to_tensor([pair[0] for pair in paired_data])
query2_tensors = tf.convert_to_tensor([pair[1] for pair in paired_data])
labels_tensors = tf.convert_to_tensor(labels)

# Convert the data to numpy arrays for splitting
query1_numpy = np.array([item.numpy() for item in query1_tensors])
query2_numpy = np.array([item.numpy() for item in query2_tensors])
labels_numpy = labels_tensors.numpy()

# Split the data
X_train_1, X_val_1, X_train_2, X_val_2, Y_train, Y_val = train_test_split(
    query1_numpy, query2_numpy, labels_numpy, test_size=0.3, random_state=42
)

# Convert back to tensors
X_train_1 = tf.convert_to_tensor(X_train_1, dtype=tf.int32)
X_val_1 = tf.convert_to_tensor(X_val_1, dtype=tf.int32)
X_train_2 = tf.convert_to_tensor(X_train_2, dtype=tf.int32)
X_val_2 = tf.convert_to_tensor(X_val_2, dtype=tf.int32)
Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
Y_val = tf.convert_to_tensor(Y_val, dtype=tf.float32)

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

# Train the model
history = siamese_model.fit(
    [X_train_1, X_train_2],
    Y_train,
    validation_data=([X_val_1, X_val_2], Y_val),
    epochs=10,
    batch_size=16
)

# Save the model
siamese_model.save('siamese_model.h5')