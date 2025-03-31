
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# -------------------- Generator --------------------
class Generator(tf.keras.Model):
    def __init__(self, noise_dim=100, output_dim=128):
        super(Generator, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(output_dim, activation='tanh')

    def call(self, z):
        x = self.dense1(z)
        x = self.bn1(x)
        return self.dense2(x)

# -------------------- Discriminator --------------------
class Discriminator(tf.keras.Model):
    def __init__(self, input_dim=128):
        super(Discriminator, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)

# -------------------- Co-Attention --------------------
class CoAttention(tf.keras.layers.Layer):
    def call(self, H_real, H_fake):
        attention_scores = tf.matmul(H_real, H_fake, transpose_b=True)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        context = tf.matmul(attention_weights, H_fake)
        return H_real + context

# -------------------- Full Model Builder --------------------
def build_bert_gan(input_shape, noise_dim=100, gru_units=64):
    input_ids = tf.keras.Input(shape=input_shape, dtype=tf.int32, name="input_ids")
    z_input = tf.keras.Input(shape=(noise_dim,), name="noise_input")

    bert_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
    bert_output = bert_model(input_ids)[0]

    gru_out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_units, return_sequences=True))(bert_output)

    generator = Generator(noise_dim=noise_dim, output_dim=gru_units * 2)
    fake_features_flat = generator(z_input)
    fake_features = tf.keras.layers.Reshape((1, gru_units * 2))(fake_features_flat)
    fake_features = tf.keras.layers.UpSampling1D(size=128)(fake_features)

    discriminator = Discriminator(input_dim=gru_units * 2)
    coattn = CoAttention()
    merged = coattn(gru_out, fake_features)

    pooled = tf.keras.layers.GlobalMaxPooling1D()(merged)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='clf_output')(pooled)

    model = tf.keras.Model(inputs=[input_ids, z_input], outputs=output)
    return model

# -------------------- Dataset Preparation --------------------
data = pd.read_csv('data.csv')
data = data.dropna()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
inputs = tokenizer(data['url'].tolist(), max_length=128, truncation=True, padding='max_length', return_tensors='tf')

X = inputs['input_ids'].numpy()
y = data['label'].map({'bad': 0, 'good': 1}).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train = tf.convert_to_tensor(X_train, dtype=tf.int32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.int32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# -------------------- Build and Train --------------------
model = build_bert_gan((128,))
model.compile(optimizer=tf.keras.optimizers.Adam(3e-6), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit([X_train, tf.random.normal((len(X_train), 100))], y_train, validation_split=0.1, epochs=5, batch_size=32, class_weight=class_weights_dict)

# -------------------- Evaluation --------------------
pred_probs = model.predict([X_test, tf.random.normal((len(X_test), 100))]).flatten()
preds = (pred_probs > 0.5).astype(int)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)
auc = roc_auc_score(y_test, pred_probs)
print(f"Accuracy: {acc:.2f}, F1: {f1:.2f}, AUC: {auc:.2f}")
