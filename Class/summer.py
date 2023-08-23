from transformers import TFBertModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class BertWithMCDO(keras.Model):
    def __init__(self, dropout_rate=0.1, num_labels=3):
        super(BertWithMCDO, self).__init__()
        self.bert = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        self.dropout_bert = layers.Dropout(dropout_rate)
        self.fc1 = layers.Dense(512, activation='relu')  # Adding the first Dense layer
        self.dropout_fc1 = layers.Dropout(dropout_rate)
        self.fc2 = layers.Dense(256, activation='relu')  # Adding the second Dense layer
        self.dropout_fc2 = layers.Dropout(dropout_rate)
        self.fc3 = layers.Dense(num_labels)  # Adding the third Dense layer

    def load(self, path):
        self.load_weights(path)

    def save(self, path):
        self.save_weights(path)

    def call(self, inputs, training=False):
        input_ids, attention_mask = inputs
        bert_output = self.bert([input_ids, attention_mask], training=training)
        pooled_output = bert_output[1]  # Using the pooled_output from BERT
        dropout_bert_output = self.dropout_bert(pooled_output, training=training)
        fc1_output = self.fc1(dropout_bert_output)
        dropout_fc1_output = self.dropout_fc1(fc1_output, training=training)
        fc2_output = self.fc2(dropout_fc1_output)
        dropout_fc2_output = self.dropout_fc2(fc2_output, training=training)
        fc3_output = self.fc3(dropout_fc2_output)
        return fc3_output

# Create an instance of the model
model = BertWithMCDO()

# Compile the model
model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

dummy_input = [tf.constant([[1, 2, 3]]), tf.constant([[1, 1, 1]])]

# Call the model to build it
_ = model(dummy_input)

# Summary of the model architecture
model.summary()
