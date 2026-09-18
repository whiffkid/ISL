import os
import sys
import glob

# Ensure matching local venv site-packages are accessible
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
for venv_site in glob.glob(os.path.join(base_dir, "venv*", "lib*", py_ver, "site-packages")):
    if venv_site not in sys.path and os.path.exists(venv_site):
        sys.path.insert(0, venv_site)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json

def get_actions(data_dir=None, model_name=None):
    """Dynamically find actions from model metadata, data folders, or fallback to pretrained class list."""
    if model_name:
        model_classes_path = os.path.join('models', model_name, 'classes.json')
        if os.path.exists(model_classes_path):
            try:
                with open(model_classes_path, 'r') as f:
                    actions = json.load(f)
                    if actions:
                        return actions
            except Exception:
                pass

    search_paths = [data_dir, 'keypoint_data', 'greetings_data']
    for path in search_paths:
        if path and os.path.exists(path):
            actions = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and not d.startswith('.')])
            if actions:
                return actions
    # Fallback default ISL actions matching pre-trained models
    return ['alright', 'good afternoon', 'good evening', 'good morning', 'good night', 'hello', 'how are you', 'pleased', 'thank you']


def lstm_v1(device_name=None, num_classes=None):
    if num_classes is None:
        num_classes = len(get_actions())

    model = keras.Sequential([
        layers.Input(shape=(30, 150)),
        layers.LSTM(64, return_sequences=True, activation='relu'),
        layers.LSTM(128, return_sequences=True, activation='relu'),
        layers.LSTM(128, return_sequences=True, activation='relu'),
        layers.LSTM(256, return_sequences=False, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    if device_name:
        with tf.device(device_name):
            return model
    return model


def lstm_v2(device_name=None, num_classes=None):
    if num_classes is None:
        num_classes = len(get_actions())

    model = keras.Sequential([
        layers.Input(shape=(30, 150)),
        layers.LSTM(64, return_sequences=True, activation='relu'),
        layers.LSTM(128, return_sequences=True, activation='relu'),
        layers.LSTM(128, return_sequences=True, activation='relu'),
        layers.LSTM(256, return_sequences=True, activation='relu'),
        layers.LSTM(256, return_sequences=False, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    if device_name:
        with tf.device(device_name):
            return model
    return model


def lstm_v3(device_name=None, num_classes=None):
    if num_classes is None:
        num_classes = len(get_actions())

    model = keras.Sequential([
        layers.Input(shape=(30, 150)),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(256, return_sequences=True),
        layers.LSTM(128, return_sequences=False),
        layers.Dense(1024, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    if device_name:
        with tf.device(device_name):
            return model
    return model


def bilstm_attention(input_shape=(30, 150), num_classes=None, device_name=None):
    """Bidirectional LSTM with Multi-Head Self-Attention for Temporal Sign Recognition."""
    if num_classes is None:
        num_classes = len(get_actions())

    inputs = keras.Input(shape=input_shape)
    
    # Feature projection & initial normalization
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.15)(x)

    # 1st Bidirectional LSTM layer
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)

    # 2nd Bidirectional LSTM layer
    lstm_out = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    
    # Multi-Head Attention over temporal features
    attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=64)(lstm_out, lstm_out)
    x = layers.Add()([lstm_out, attn_out])
    x = layers.LayerNormalization()(x)
    
    # Global pooling over sequence length
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])

    # Classification head
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="bilstm_attention")
    if device_name:
        with tf.device(device_name):
            return model
    return model


@tf.keras.utils.register_keras_serializable(package="Custom")
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length=30, d_model=128, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.pos_embedding = layers.Embedding(input_dim=sequence_length, output_dim=d_model)

    def build(self, input_shape):
        self.pos_embedding.build((self.sequence_length,))
        super().build(input_shape)

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        return inputs + self.pos_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
        })
        return config


def transformer(
        input_shape=(30, 150),
        output_shape=None,
        head_size=128,
        num_heads=4,
        ff_dim=256,
        num_transformer_blocks=3,
        mlp_units=[256, 128],
        dropout=0.15,
        mlp_dropout=0.2,
        device_name=None
):
    """Transformer Encoder Model for Skeleton Keypoint Sequence Classification."""
    if output_shape is None:
        output_shape = len(get_actions())

    inputs = keras.Input(shape=input_shape)
    
    # Project input features to embedding dimension
    x = layers.Dense(head_size, activation="relu")(inputs)
    x = PositionalEmbedding(sequence_length=input_shape[0], d_model=head_size)(x)
    x = layers.Dropout(dropout)(x)

    # Stack Transformer Encoder blocks
    for _ in range(num_transformer_blocks):
        # Attention block
        norm_1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size // num_heads, dropout=dropout)(norm_1, norm_1)
        x = layers.Add()([x, attn])

        # Feed Forward block
        norm_2 = layers.LayerNormalization(epsilon=1e-6)(x)
        ff = layers.Dense(ff_dim, activation="relu")(norm_2)
        ff = layers.Dropout(dropout)(ff)
        ff = layers.Dense(head_size)(ff)
        x = layers.Add()([x, ff])

    x = layers.GlobalAveragePooling1D()(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(output_shape, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="transformer")
    if device_name:
        with tf.device(device_name):
            return model
    return model


def compile_model(model):
    adam = keras.optimizers.Adam(learning_rate=3e-4)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def build_raw_model(name, device=None, num_classes=None):
    if name == 'lstm_v1':
        return lstm_v1(device, num_classes)
    elif name == 'lstm_v2':
        return lstm_v2(device, num_classes)
    elif name == 'lstm_v3':
        return lstm_v3(device, num_classes)
    elif name == 'bilstm_attention':
        return bilstm_attention(device_name=device, num_classes=num_classes)
    elif name == 'transformer':
        return transformer(output_shape=num_classes, device_name=device)
    else:
        raise ValueError(f"Unknown model architecture: {name}")


def load_model(name='lstm_v3', pretrained=False, training=True, device=None, num_classes=None):
    if num_classes is None:
        num_classes = len(get_actions(model_name=name))

    if not pretrained:
        model = build_raw_model(name, device, num_classes=num_classes)
        if training:
            return compile_model(model)
        return model

    model_dir = os.path.join('models', name)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found.")

    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.keras') or f.endswith('.h5')]
    if not model_files:
        raise FileNotFoundError(f"No model file found in '{model_dir}'.")

    # Prefer native .keras format if present, else .h5
    model_files.sort(key=lambda x: 0 if x.endswith('.keras') else 1)
    model_path = model_files[0]
    print(f"Loading Model from : {model_path}")

    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={'PositionalEmbedding': PositionalEmbedding}
        )
    except Exception as err:
        print(f"Direct Keras load failed ({err}). Loading weights onto architecture...")
        model = build_raw_model(name, device, num_classes=num_classes)
        model.load_weights(model_path)

    if training:
        return compile_model(model)
    return model