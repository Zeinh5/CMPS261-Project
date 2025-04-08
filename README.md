# CMPS261-Project
Sentiment Analysis for Mental health: Develop a machine learning model capable of accurately classifying text statements into the seven mental  health categories.

write this cell when to load and use the model:
# Load tokenizer and max_len

<!-- with open('preprocessing.pkl', 'rb') as f:
    preprocessing_data = pickle.load(f)
tokenizer = preprocessing_data['tokenizer']
max_len = preprocessing_data['max_len']

# Load label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the trained model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Embedding # Import necessary classes

# Define custom layers
class AttentionPooling(Layer): # Defining the custom layer inside the loading script
    def __init__(self, units, **kwargs): # Added **kwargs
        super().__init__(**kwargs) # Pass **kwargs to super().__init__
        self.W = Dense(units, activation='tanh')
        self.V = Dense(1)

    def call(self, features):
        score = self.V(self.W(features))
        weights = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(weights * features, axis=1)

class PositionalEmbedding(Layer): # Defining the custom layer inside the loading script
    def __init__(self, vocab_size, embed_dim, maxlen, **kwargs): # Added **kwargs to accept extra arguments
        super().__init__(**kwargs) # Pass extra arguments to super().__init__
        self.token_embed = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_embed = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        length = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        return self.token_embed(x) + self.pos_embed(positions)

# Load the model with custom_objects
model = load_model('sentiment_model.h5', custom_objects={'PositionalEmbedding': PositionalEmbedding, 'AttentionPooling': AttentionPooling}) -->
