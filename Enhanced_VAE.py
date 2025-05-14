import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers, models, callbacks, regularizers, backend as K

# Import VGG16 specifically

from tensorflow.keras.applications import VGG16

import rasterio # For reading TIFF files

import glob

import os

import time

from tqdm.notebook import tqdm # Progress bar for loading

import matplotlib.pyplot as plt

from google.colab import drive

import traceback

from sklearn.model_selection import train_test_split

  

# Record start time

start_time = time.time()

  

# --- Configuration ---

DRIVE_PATH = '/content/drive/My Drive/ndvi_rnd/'

FOREST_IMG_DIR = DRIVE_PATH + 'ndvi_rnd/Forest/' # Ensure this path is correct

PREPROCESSED_DATA_PATH = DRIVE_PATH + 'vae_forest_preprocessed_data.npy' # Path for NumPy array

IMG_HEIGHT = 64

IMG_WIDTH = 64

IMG_CHANNELS = 13

LATENT_DIM = 256

L1_LOSS_WEIGHT = 1.0       # Weight for pixel-wise L1 loss

PERCEPTUAL_LOSS_WEIGHT = 0.1 # Weight for VGG perceptual loss (start smaller)

KL_WEIGHT = 0.005           # Keep KL weight relatively small initially

LEARNING_RATE = 1e-4

EPOCHS = 200

BATCH_SIZE = 32

EARLY_STOPPING_PATIENCE = 25

NORM_MODE = 'scale'

NORM_DIVISOR = 10000.0

MODEL_SAVE_PATH = DRIVE_PATH + 'vae_forest_s2_perc_L1_v1.h5' # Indicate Perceptual+L1

CHECKPOINT_DIR = DRIVE_PATH + 'checkpoints_vae_forest_perc_L1_v1/'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

  

# *** NDVI Configuration ***

# !!! IMPORTANT: Verify these indices for your 13-channel data !!!

# Assuming standard Sentinel-2 L2A band order subset:

# B2=1, B3=2, B4=3(Red), B8=7(NIR), B11=10, B12=11 ... (0-indexed)

NDVI_RED_INDEX = 3

NDVI_NIR_INDEX = 7

# Bands for VGG input (visualisation will also use these)

VGG_BANDS_INDICES = [NDVI_RED_INDEX, 2, 1] # R, G, B -> B4, B3, B2

# *************************

  

# --- Mount Drive ---

print("Mounting Google Drive...")

try:

    drive.mount('/content/drive', force_remount=True)

    print("Drive mounted successfully.")

except Exception as e: print(f"Error mounting drive: {e}")

  

# --- Data Loading and Preprocessing Function (Unchanged) ---

def load_and_preprocess_tiffs(directory_path, img_height, img_width, img_channels, norm_mode='scale', norm_divisor=10000.0):

    print(f"Searching for .tif files recursively in: {directory_path}")

    filepaths = glob.glob(os.path.join(directory_path, '**/*.tif'), recursive=True)

    if not filepaths:

        if not os.path.isdir(directory_path): raise ValueError(f"Directory not found: {directory_path}.")

        else: raise ValueError(f"No .tif files found recursively in directory: {directory_path}")

    print(f"Found {len(filepaths)} .tif files.")

    all_images = []; valid_images_count = 0; skipped_count = 0

    for i, fp in enumerate(tqdm(filepaths, desc="Loading Images")):

        img = None

        try:

            with rasterio.open(fp) as src:

                img = src.read().transpose((1, 2, 0))

                if img.ndim != 3: print(f"[{i+1}/{len(filepaths)}] Skipping {fp}: Dims {img.shape}"); skipped_count += 1; continue

                if img.shape[0]!=img_height or img.shape[1]!=img_width:

                     img_tensor=tf.constant(img, dtype=tf.float32); img_resized=tf.image.resize(tf.expand_dims(img_tensor, axis=0), [img_height, img_width], method='bilinear'); img=tf.squeeze(img_resized, axis=0).numpy()

                if img.shape[2] != img_channels: print(f"[{i+1}/{len(filepaths)}] Skipping {fp}: Channels {img.shape[2]}"); skipped_count += 1; continue

                if np.isnan(img).any() or np.isinf(img).any(): img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

                if isinstance(img, np.ndarray) and img.shape==(img_height, img_width, img_channels): all_images.append(img); valid_images_count += 1

                else: print(f"[{i+1}/{len(filepaths)}] Skipping {fp}: Final check fail. Shape={img.shape if hasattr(img,'shape') else 'N/A'}, Type={type(img)}"); skipped_count += 1; continue

        except Exception as e: print(f"!!! [{i+1}/{len(filepaths)}] ERROR file {fp}: {e} !!!"); traceback.print_exc(); skipped_count += 1

    print(f"\nFinished loading. Processed: {valid_images_count}, Skipped/Errored: {skipped_count}")

    if not all_images: raise ValueError("No images loaded.")

    num_non_arrays = sum(1 for item in all_images if not isinstance(item, np.ndarray))

    if num_non_arrays > 0: raise TypeError("Not all items are NumPy arrays.")

    print("Converting to NumPy array..."); data = np.array(all_images, dtype=np.float32); print(f"Data shape: {data.shape}")

    print(f"Normalizing: {norm_mode}"); data = data / norm_divisor; data = np.clip(data, 0.0, 1.0); print(f"Scaled data range: {data.min()} - {data.max()}")

    return data

  

# --- NDVI Calculation Function ---

def calculate_ndvi(image_data, nir_index, red_index, epsilon=1e-8):

    """Calculates NDVI from a single image array (H, W, C)."""

    # Ensure indices are valid

    if nir_index >= image_data.shape[-1] or red_index >= image_data.shape[-1]:

        raise IndexError("NIR or Red index out of bounds for image channels.")

    # Extract bands - handle potential scaling if needed (assuming input is [0,1])

    nir = image_data[:, :, nir_index]

    red = image_data[:, :, red_index]

    # Calculate NDVI

    numerator = nir - red

    denominator = nir + red + epsilon # Add epsilon to avoid division by zero

    ndvi = numerator / denominator

    # Clip NDVI to the theoretical range [-1, 1]

    ndvi = np.clip(ndvi, -1.0, 1.0)

    return ndvi

  

# --- Load Data (Check for .npy first) and Create Datasets (Unchanged logic)---

# ... (Keep the loading logic checking for PREPROCESSED_DATA_PATH first) ...

train_dataset = None; val_dataset = None; val_data = None; forest_data = None

print("\n--- Checking for Preprocessed Data ---")

if os.path.exists(PREPROCESSED_DATA_PATH):

    try:

        print(f"Loading preprocessed data from: {PREPROCESSED_DATA_PATH}")

        forest_data = np.load(PREPROCESSED_DATA_PATH)

        print(f"Successfully loaded preprocessed data with shape: {forest_data.shape}")

        if forest_data.ndim != 4 or forest_data.shape[1:] != (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS): print(f"Warning: Loaded data shape {forest_data.shape} mismatch. Reloading."); forest_data = None

        elif len(forest_data) == 0: print("Warning: Loaded file empty. Reloading."); forest_data = None

    except Exception as e: print(f"Error loading {PREPROCESSED_DATA_PATH}: {e}. Reloading."); forest_data = None

if forest_data is None:

    try:

        print("\n--- Starting Data Loading from TIFFs ---")

        if not os.path.isdir(FOREST_IMG_DIR): print(f"ERROR: TIFF directory not found: {FOREST_IMG_DIR}"); exit()

        forest_data = load_and_preprocess_tiffs(FOREST_IMG_DIR, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, NORM_MODE, NORM_DIVISOR)

        if forest_data is not None and len(forest_data) > 0:

            try: print(f"\n--- Saving Preprocessed Data to: {PREPROCESSED_DATA_PATH} ---"); np.save(PREPROCESSED_DATA_PATH, forest_data); print("Preprocessed data saved successfully.")

            except Exception as e: print(f"!!! Warning: Failed to save preprocessed data: {e} !!!")

        else: print("ERROR: Forest data loading resulted in empty or None data."); exit()

    except Exception as e: print(f"\n!!! ERROR during TIFF loading/saving: {e} !!!"); traceback.print_exc(); exit()

if forest_data is not None and len(forest_data) > 0:

    print("\n--- Splitting Data ---")

    train_data, val_data_split = train_test_split(forest_data, test_size=0.1, random_state=42)

    val_data = val_data_split

    print(f"Train shape: {train_data.shape}, Val shape: {val_data.shape}")

    if len(train_data) == 0 or len(val_data) == 0: print("ERROR: Zero samples in split."); exit()

    print("\n--- Creating tf.data Datasets ---")

    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(len(train_data)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(val_data).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print("Datasets created.")

else: print("ERROR: No valid data available to create datasets."); exit()

  

# --- FULL Dataset Checks (Unchanged) ---

# ... (Keep the full dataset integrity check code) ...

print("\nPerforming FULL dataset integrity check before training...")

print("Checking train_dataset...")

if train_dataset is None: print("ERROR: train_dataset is None."); exit()

try:

    batch_num = 0; sample_train_batch = None

    for batch in tqdm(train_dataset, desc="Iterating train_dataset"):

        if batch_num == 0: sample_train_batch = batch

        if batch is None: print(f"ERROR: Found None batch {batch_num}."); exit()

        batch_data = batch[0] if isinstance(batch, tuple) else batch

        if not isinstance(batch_data, tf.Tensor): print(f"ERROR: Batch {batch_num} not Tensor."); exit()

        if len(tf.shape(batch_data)) != 4: print(f"ERROR: Batch {batch_num} wrong rank."); exit()

        batch_num += 1

    if sample_train_batch is not None: print(f"Sample train batch shape: {tf.shape(sample_train_batch[0] if isinstance(sample_train_batch, tuple) else sample_train_batch)}")

    print(f"Train check passed ({batch_num} batches).")

except Exception as e: print(f"!!! Error checking train_dataset (batch {batch_num}) !!!"); print(f"Error: {e}"); traceback.print_exc(); exit()

print("\nChecking val_dataset...")

if val_dataset is None: print("ERROR: val_dataset is None."); exit()

try:

    batch_num = 0; sample_val_batch = None

    for batch in tqdm(val_dataset, desc="Iterating val_dataset"):

        if batch_num == 0: sample_val_batch = batch

        if batch is None: print(f"ERROR: Found None batch {batch_num}."); exit()

        batch_data = batch[0] if isinstance(batch, tuple) else batch

        if not isinstance(batch_data, tf.Tensor): print(f"ERROR: Batch {batch_num} not Tensor."); exit()

        if len(tf.shape(batch_data)) != 4: print(f"ERROR: Batch {batch_num} wrong rank."); exit()

        batch_num += 1

    if sample_val_batch is not None: print(f"Sample validation batch shape: {tf.shape(sample_val_batch[0] if isinstance(sample_val_batch, tuple) else sample_val_batch)}")

    print(f"Validation check passed ({batch_num} batches).")

except Exception as e: print(f"!!! Error checking val_dataset (batch {batch_num}) !!!"); print(f"Error: {e}"); traceback.print_exc(); exit()

  

# --- VAE Model Components (Unchanged Architectures) ---

class Sampling(layers.Layer):

    def call(self, inputs): z_mean, z_log_var = inputs; batch = tf.shape(z_mean)[0]; dim = tf.shape(z_mean)[1]; epsilon = K.random_normal(shape=(batch, dim)); return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self): return super().get_config()

def build_encoder(input_shape, latent_dim):

    encoder_inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(64, 3, strides=1, padding="same")(encoder_inputs); x = layers.LeakyReLU(alpha=0.2)(x); x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same")(x); x = layers.LeakyReLU(alpha=0.2)(x); x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, 3, strides=2, padding="same")(x); x = layers.LeakyReLU(alpha=0.2)(x); x = layers.BatchNormalization()(x)

    x = layers.Conv2D(512, 3, strides=2, padding="same")(x); x = layers.LeakyReLU(alpha=0.2)(x); x = layers.BatchNormalization()(x)

    x = layers.Conv2D(512, 3, strides=2, padding="same")(x); x = layers.LeakyReLU(alpha=0.2)(x); x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)

    x = layers.Dense(1024, activation="relu")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x); z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    return keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")

def build_decoder(latent_dim, encoder_concrete_output_shape, output_channels):

    latent_inputs = keras.Input(shape=(latent_dim,))

    dense_units = np.prod(encoder_concrete_output_shape)

    x = layers.Dense(dense_units)(latent_inputs); x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Reshape(encoder_concrete_output_shape)(x)

    x = layers.UpSampling2D(size=(2, 2))(x); x = layers.Conv2D(512, 3, padding="same")(x); x = layers.LeakyReLU(alpha=0.2)(x); x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(size=(2, 2))(x); x = layers.Conv2D(256, 3, padding="same")(x); x = layers.LeakyReLU(alpha=0.2)(x); x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(size=(2, 2))(x); x = layers.Conv2D(128, 3, padding="same")(x); x = layers.LeakyReLU(alpha=0.2)(x); x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(size=(2, 2))(x); x = layers.Conv2D(64, 3, padding="same")(x); x = layers.LeakyReLU(alpha=0.2)(x); x = layers.BatchNormalization()(x)

    decoder_outputs = layers.Conv2D(output_channels, 3, activation="sigmoid", padding="same")(x)

    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

def build_vgg_feature_extractor(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):

    vgg = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    vgg.trainable = False

    layer_names = ["block1_conv2", "block2_conv2", "block3_conv3"]

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = keras.Model(inputs=vgg.input, outputs=outputs, name="vgg_feature_extractor")

    return model

  

# --- VAE Class with Perceptual Loss and Corrected L1 Axis Sum ---

class VAE_Perceptual(keras.Model):

    def __init__(self, encoder, decoder, vgg_feature_extractor, kl_weight=0.01, l1_weight=1.0, perceptual_weight=0.1, vgg_input_indices=[3,2,1], **kwargs):

        super().__init__(**kwargs)

        self.encoder = encoder; self.decoder = decoder; self.vgg = vgg_feature_extractor

        self.sampling = Sampling(); self.kl_weight = kl_weight; self.l1_weight = l1_weight

        self.perceptual_weight = perceptual_weight; self.vgg_input_indices = vgg_input_indices

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")

        self.pixel_loss_tracker = keras.metrics.Mean(name="pixel_l1_loss")

        self.perceptual_loss_tracker = keras.metrics.Mean(name="perceptual_loss")

        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        self.l1_loss_fn = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

        self.feature_loss_fn = tf.keras.losses.MeanSquaredError()

    @property

    def metrics(self): return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.pixel_loss_tracker, self.perceptual_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs): z_mean, z_log_var = self.encoder(inputs); z = self.sampling([z_mean, z_log_var]); return self.decoder(z)

    def _adapt_for_vgg(self, images):

        images_rgb = tf.gather(images, self.vgg_input_indices, axis=-1)

        images_rgb_scaled = images_rgb * 255.0

        return tf.keras.applications.vgg16.preprocess_input(images_rgb_scaled)

    def _calculate_perceptual_loss(self, y_true, y_pred):

        y_true_vgg = self._adapt_for_vgg(y_true); y_pred_vgg = self._adapt_for_vgg(y_pred)

        true_features = self.vgg(y_true_vgg, training=False); pred_features = self.vgg(y_pred_vgg, training=False)

        total_perceptual_loss = 0.0

        if not isinstance(true_features, list): true_features = [true_features]

        if not isinstance(pred_features, list): pred_features = [pred_features]

        for true_feat, pred_feat in zip(true_features, pred_features):

            total_perceptual_loss += self.feature_loss_fn(true_feat, pred_feat)

        return total_perceptual_loss / max(1, len(true_features))

    def train_step(self, data):

        if isinstance(data, tuple): data = data[0]

        with tf.GradientTape() as tape:

            z_mean, z_log_var = self.encoder(data); z = self.sampling([z_mean, z_log_var]); reconstruction = self.decoder(z)

            per_pixel_l1_loss = self.l1_loss_fn(data, reconstruction) # Shape: (batch, H, W) after MAE averages channel

            pixel_l1_loss_per_sample = tf.reduce_sum(per_pixel_l1_loss, axis=(1, 2)) # Sum over H, W

            pixel_l1_loss = tf.reduce_mean(pixel_l1_loss_per_sample)

            perceptual_loss = self._calculate_perceptual_loss(data, reconstruction)

            reconstruction_loss = (self.l1_weight * pixel_l1_loss) + (self.perceptual_weight * perceptual_loss)

            kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1))

            total_loss = reconstruction_loss + self.kl_weight * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights); self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss); self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        self.pixel_loss_tracker.update_state(pixel_l1_loss); self.perceptual_loss_tracker.update_state(perceptual_loss); self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):

        if isinstance(data, tuple): data = data[0]

        z_mean, z_log_var = self.encoder(data); z = self.sampling([z_mean, z_log_var]); reconstruction = self.decoder(z)

        per_pixel_l1_loss = self.l1_loss_fn(data, reconstruction) # Shape: (batch, H, W)

        pixel_l1_loss_per_sample = tf.reduce_sum(per_pixel_l1_loss, axis=(1, 2)) # Sum over H, W

        pixel_l1_loss = tf.reduce_mean(pixel_l1_loss_per_sample)

        perceptual_loss = self._calculate_perceptual_loss(data, reconstruction)

        reconstruction_loss = (self.l1_weight * pixel_l1_loss) + (self.perceptual_weight * perceptual_loss)

        kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1))

        total_loss = reconstruction_loss + self.kl_weight * kl_loss

        self.total_loss_tracker.update_state(total_loss); self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        self.pixel_loss_tracker.update_state(pixel_l1_loss); self.perceptual_loss_tracker.update_state(perceptual_loss); self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self): base_config = super().get_config(); config = { "encoder": keras.saving.serialize_keras_object(self.encoder), "decoder": keras.saving.serialize_keras_object(self.decoder), "kl_weight": self.kl_weight, "l1_weight": self.l1_weight, "perceptual_weight": self.perceptual_weight, "vgg_input_indices": self.vgg_input_indices, }; return {**base_config, **config}

    @classmethod

    def from_config(cls, config):

        encoder_config = config.pop("encoder"); decoder_config = config.pop("decoder")

        encoder = keras.saving.deserialize_keras_object(encoder_config); decoder = keras.saving.deserialize_keras_object(decoder_config)

        vgg_input_indices = config.get("vgg_input_indices", [3, 2, 1]); vgg_input_shape = (IMG_HEIGHT, IMG_WIDTH, len(vgg_input_indices))

        vgg_feature_extractor = build_vgg_feature_extractor(input_shape=vgg_input_shape)

        instance = cls(encoder, decoder, vgg_feature_extractor, **config)

        instance.l1_loss_fn = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

        instance.feature_loss_fn = tf.keras.losses.MeanSquaredError()

        return instance

  

# --- Build and Compile VAE ---

print("\nBuilding Enhanced VAE with Perceptual Loss...")

input_vae_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

encoder_model = build_encoder(input_vae_shape, LATENT_DIM)

layer_before_flatten = None; flatten_layer_found = False

for layer in encoder_model.layers:

    if isinstance(layer, layers.Flatten): flatten_layer_found = True; break

    layer_before_flatten = layer

if not flatten_layer_found or layer_before_flatten is None: raise ValueError("Could not find Flatten layer.")

encoder_concrete_output_shape = layer_before_flatten.output.shape[1:]

encoder_concrete_output_shape = tuple(dim if dim is not None else -1 for dim in encoder_concrete_output_shape)

if -1 in encoder_concrete_output_shape: print(f"Warning: Dynamic shape {encoder_concrete_output_shape}. Using fallback (4, 4, 512)."); encoder_concrete_output_shape = (4, 4, 512)

print(f"Determined encoder concrete output shape: {encoder_concrete_output_shape}")

decoder_model = build_decoder(LATENT_DIM, encoder_concrete_output_shape, IMG_CHANNELS)

vgg_input_shape = (IMG_HEIGHT, IMG_WIDTH, len(VGG_BANDS_INDICES)) # Use defined indices

vgg_feature_extractor = build_vgg_feature_extractor(input_shape=vgg_input_shape)

print("VGG Feature Extractor built.")

vae = VAE_Perceptual(encoder=encoder_model, decoder=decoder_model, vgg_feature_extractor=vgg_feature_extractor, kl_weight=KL_WEIGHT, l1_weight=L1_LOSS_WEIGHT, perceptual_weight=PERCEPTUAL_LOSS_WEIGHT, vgg_input_indices=VGG_BANDS_INDICES)

build_input_shape = (None,) + input_vae_shape

vae.build(build_input_shape)

print(f"VAE model explicitly built with input shape: {build_input_shape}")

print("\nEncoder Summary:"); vae.encoder.summary(line_length=100)

print("\nDecoder Summary:"); vae.decoder.summary(line_length=100)

print("\nCompiling Enhanced VAE with Perceptual Loss...")

vae.compile(optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-5))

  

# --- Callbacks ---

print("\nSetting up callbacks...")

checkpoint_filepath = CHECKPOINT_DIR + 'vae_epoch_{epoch:02d}-val_rec_loss_{val_reconstruction_loss:.4f}.weights.h5'

vae_checkpoint_callback = callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='val_reconstruction_loss', mode='min', save_best_only=True, verbose=1)

vae_early_stopping = callbacks.EarlyStopping(monitor='val_reconstruction_loss', patience=EARLY_STOPPING_PATIENCE, mode='min', verbose=1, restore_best_weights=True)

vae_reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_reconstruction_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1)

vae_callbacks_list = [vae_early_stopping, vae_reduce_lr, vae_checkpoint_callback]

  

# --- Train VAE ---

print("\nStarting Enhanced VAE training (Perceptual + L1 Loss)...")

history = vae.fit(

    train_dataset,

    epochs=EPOCHS,

    validation_data=val_dataset,

    callbacks=vae_callbacks_list,

    verbose=1

)

print("VAE Training finished.")

  

# --- Save Final Model ---

print(f"\nSaving final Enhanced VAE model (best weights) to {MODEL_SAVE_PATH}...")

try:

    vae.save(MODEL_SAVE_PATH)

    print("VAE model saved successfully.")

except Exception as e:

    print(f"Error saving full VAE model: {e}")

    print("Saving weights only as fallback...")

    try: vae.save_weights(MODEL_SAVE_PATH.replace('.h5', '.weights.h5')); print("VAE weights saved successfully.")

    except Exception as e2: print(f"Error saving VAE weights: {e2}")

  

# --- Evaluation and Visualization (with NDVI) ---

print("\n--- Evaluating Enhanced VAE ---")

# Plot Losses

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

plt.plot(history.history.get('total_loss', []), label='Train Total Loss'); plt.plot(history.history.get('val_total_loss', []), label='Val Total Loss')

plt.plot(history.history.get('reconstruction_loss', []), label='Train Rec Loss (Comb)', linestyle=':'); plt.plot(history.history.get('val_reconstruction_loss', []), label='Val Rec Loss (Comb)', linestyle=':')

plt.plot(history.history.get('kl_loss', []), label='Train KL Loss', linestyle='-.'); plt.plot(history.history.get('val_kl_loss', []), label='Val KL Loss', linestyle='-.')

plt.title('Enhanced VAE Total & KL Loss'); plt.ylabel('Loss'); plt.legend(); plt.grid(True); plt.ylim(bottom=0)

plt.subplot(2, 1, 2)

plt.plot(history.history.get('pixel_l1_loss', []), label='Train Pixel L1 Loss'); plt.plot(history.history.get('val_pixel_l1_loss', []), label='Val Pixel L1 Loss')

plt.plot(history.history.get('perceptual_loss', []), label='Train Perceptual Loss', linestyle='--'); plt.plot(history.history.get('val_perceptual_loss', []), label='Val Perceptual Loss', linestyle='--')

plt.title('Enhanced VAE Reconstruction Loss Components'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True); plt.ylim(bottom=0)

plt.tight_layout(); plt.show()

  

# Visualize Reconstructions and NDVI

if val_data is not None and len(val_data) > 0:

    n_show = min(8, len(val_data)) # Show slightly fewer to fit NDVI

    indices = np.random.choice(len(val_data), n_show, replace=False)

    original_images = val_data[indices]

  

    print("Generating reconstructed images...")

    reconstructed_images = vae.predict(original_images, batch_size=BATCH_SIZE)

  

    print(f"\nShowing {n_show} original vs reconstructed images and NDVI...")

    vis_bands_indices = VGG_BANDS_INDICES # Use RGB bands for visual check

  

    # Store NDVI diffs

    all_ndvi_orig = []

    all_ndvi_recon = []

  

    plt.figure(figsize=(n_show * 2.5, 10)) # Adjust figure size for 4 rows

    for i in range(n_show):

        # --- Original Image (RGB) ---

        ax = plt.subplot(4, n_show, i + 1)

        try:

            orig_vis = original_images[i][:, :, vis_bands_indices]; p1, p99 = np.percentile(orig_vis[orig_vis > 0], (1, 99)); p99 = max(p99, p1 + 1e-6); orig_vis = np.clip((np.clip(orig_vis, p1, p99) - p1) / (p99 - p1), 0.0, 1.0); plt.imshow(orig_vis)

        except Exception as img_e: print(f"Error processing original image {i} for display: {img_e}"); plt.imshow(np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)))

        if i == 0: plt.title("Original\nRGB Vis")

        else: plt.title("Original")

        plt.axis("off")

  

        # --- Reconstructed Image (RGB) ---

        ax = plt.subplot(4, n_show, i + 1 + n_show)

        try:

            recon_vis = reconstructed_images[i][:, :, vis_bands_indices]; p1, p99 = np.percentile(recon_vis[recon_vis > 0], (1, 99)); p99 = max(p99, p1 + 1e-6); recon_vis = np.clip((np.clip(recon_vis, p1, p99) - p1) / (p99 - p1), 0.0, 1.0); plt.imshow(recon_vis)

        except Exception as img_e: print(f"Error processing reconstructed image {i} for display: {img_e}"); plt.imshow(np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)))

        if i == 0: plt.title("Reconstructed\nRGB Vis")

        else: plt.title("Reconstructed")

        plt.axis("off")

  

        # --- Original NDVI ---

        ax = plt.subplot(4, n_show, i + 1 + 2 * n_show)

        try:

            ndvi_orig = calculate_ndvi(original_images[i], NDVI_NIR_INDEX, NDVI_RED_INDEX)

            im = plt.imshow(ndvi_orig, cmap='RdYlGn', vmin=-1, vmax=1)

            all_ndvi_orig.append(ndvi_orig)

        except Exception as ndvi_e: print(f"Error calculating/plotting orig NDVI {i}: {ndvi_e}"); plt.imshow(np.zeros((IMG_HEIGHT,IMG_WIDTH)), cmap='gray')

        if i == 0: plt.title("Original\nNDVI"); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        else: plt.title("Original")

        plt.axis("off")

  

        # --- Reconstructed NDVI ---

        ax = plt.subplot(4, n_show, i + 1 + 3 * n_show)

        try:

            ndvi_recon = calculate_ndvi(reconstructed_images[i], NDVI_NIR_INDEX, NDVI_RED_INDEX)

            im = plt.imshow(ndvi_recon, cmap='RdYlGn', vmin=-1, vmax=1)

            all_ndvi_recon.append(ndvi_recon)

        except Exception as ndvi_e: print(f"Error calculating/plotting recon NDVI {i}: {ndvi_e}"); plt.imshow(np.zeros((IMG_HEIGHT,IMG_WIDTH)), cmap='gray')

        if i == 0: plt.title("Reconstructed\nNDVI"); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        else: plt.title("Reconstructed")

        plt.axis("off")

  

    plt.tight_layout()

    plt.show()

  

    # --- Quantitative NDVI Comparison ---

    if all_ndvi_orig and all_ndvi_recon and len(all_ndvi_orig) == len(all_ndvi_recon):

        all_ndvi_orig = np.array(all_ndvi_orig)

        all_ndvi_recon = np.array(all_ndvi_recon)

        ndvi_mae = np.mean(np.abs(all_ndvi_orig - all_ndvi_recon))

        ndvi_mse = np.mean((all_ndvi_orig - all_ndvi_recon)**2)

        print(f"\nNDVI Comparison (for {n_show} visualized samples):")

        print(f"  Mean Absolute Error (MAE): {ndvi_mae:.4f}")

        print(f"  Mean Squared Error (MSE):  {ndvi_mse:.4f}")

    else:

        print("\nCould not perform quantitative NDVI comparison due to errors or empty lists.")

  

else: print("Validation data not available for visualization.")

  

# End time

end_time = time.time()

print(f"\nScript finished in {end_time - start_time:.2f} seconds.")
