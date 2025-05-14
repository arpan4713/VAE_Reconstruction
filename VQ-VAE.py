# Full code with corrected plotting syntax and previous fixes

  

import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers, models, callbacks, backend as K

# Import VGG16 specifically

from tensorflow.keras.applications import VGG16

import rasterio

import glob

import os

import time

from tqdm.notebook import tqdm

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

  

# *** VQ-VAE Hyperparameters ***

EMBEDDING_DIM = 128

NUM_EMBEDDINGS = 512

COMMITMENT_BETA = 0.25

REC_LOSS_WEIGHT = 1.0 # Using standard MSE loss

# ****************************

  

LEARNING_RATE = 2e-4

EPOCHS = 200

BATCH_SIZE = 32

EARLY_STOPPING_PATIENCE = 30 # Increased patience

  

NORM_MODE = 'scale'

NORM_DIVISOR = 10000.0

  

# *** Update model save paths ***

MODEL_SAVE_PATH = DRIVE_PATH + 'vq_vae_forest_s2_v1.keras' # Use .keras format

CHECKPOINT_DIR = DRIVE_PATH + 'checkpoints_vq_vae_forest_v1/'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

  

# *** NDVI/Vis Configuration ***

NDVI_RED_INDEX = 3; NDVI_NIR_INDEX = 7; VGG_BANDS_INDICES = [NDVI_RED_INDEX, 2, 1] # R, G, B -> B4, B3, B2

  

# --- Mount Drive ---

print("Mounting Google Drive...")

try: drive.mount('/content/drive', force_remount=True); print("Drive mounted.")

except Exception as e: print(f"Error mounting drive: {e}")

  

# --- Data Loading and Preprocessing Function ---

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

    if nir_index >= image_data.shape[-1] or red_index >= image_data.shape[-1]: raise IndexError("NDVI indices out of bounds.")

    nir = image_data[:, :, nir_index]; red = image_data[:, :, red_index]

    numerator = nir - red; denominator = nir + red + epsilon

    ndvi = numerator / denominator; ndvi = np.clip(ndvi, -1.0, 1.0)

    return ndvi

  

# --- Load Data (Check for .npy first) and Create Datasets ---

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

  

# --- VQ-VAE Model Components ---

@tf.keras.utils.register_keras_serializable()

class VectorQuantizer(layers.Layer):

    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):

        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim; self.num_embeddings = num_embeddings; self.beta = beta

        w_init = tf.random_uniform_initializer()

        self.embeddings = tf.Variable( initial_value=w_init( shape=(self.embedding_dim, self.num_embeddings), dtype="float32"), trainable=True, name="embeddings_vqvae", )

    def call(self, x):

        input_shape = tf.shape(x); flattened = tf.reshape(x, [-1, self.embedding_dim])

        distances = ( tf.reduce_sum(flattened**2, axis=1, keepdims=True) - 2 * tf.matmul(flattened, self.embeddings) + tf.reduce_sum(self.embeddings**2, axis=0, keepdims=True) )

        encoding_indices = tf.argmin(distances, axis=1)

        quantized = tf.nn.embedding_lookup(tf.transpose(self.embeddings), encoding_indices)

        quantized = tf.reshape(quantized, input_shape)

        commitment_loss = self.beta * tf.reduce_mean( (tf.stop_gradient(quantized) - x) ** 2 )

        self.add_loss(commitment_loss)

        quantized = x + tf.stop_gradient(quantized - x); return quantized

    def get_config(self): config = super().get_config(); config.update({ "embedding_dim": self.embedding_dim, "num_embeddings": self.num_embeddings, "beta": self.beta, }); return config

  

def build_vq_encoder(input_shape, embedding_dim):

    encoder_inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(64, 3, strides=1, padding="same")(encoder_inputs); x = layers.LeakyReLU(alpha=0.2)(x);

    x = layers.Conv2D(128, 3, strides=2, padding="same")(x); x = layers.LeakyReLU(alpha=0.2)(x);

    x = layers.Conv2D(256, 3, strides=2, padding="same")(x); x = layers.LeakyReLU(alpha=0.2)(x);

    x = layers.Conv2D(512, 3, strides=2, padding="same")(x); x = layers.LeakyReLU(alpha=0.2)(x);

    x = layers.Conv2D(embedding_dim, 1, padding="same")(x)

    return keras.Model(encoder_inputs, x, name="encoder")

  

def build_vq_decoder(input_shape, output_channels):

    decoder_inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(512, 1, padding="same")(decoder_inputs); x = layers.LeakyReLU(alpha=0.2)(x);

    x = layers.UpSampling2D(size=(2, 2))(x); x = layers.Conv2D(512, 3, padding="same")(x); x = layers.LeakyReLU(alpha=0.2)(x);

    x = layers.UpSampling2D(size=(2, 2))(x); x = layers.Conv2D(256, 3, padding="same")(x); x = layers.LeakyReLU(alpha=0.2)(x);

    x = layers.UpSampling2D(size=(2, 2))(x); x = layers.Conv2D(128, 3, padding="same")(x); x = layers.LeakyReLU(alpha=0.2)(x);

    decoder_outputs = layers.Conv2D(output_channels, 3, activation="sigmoid", padding="same")(x)

    return keras.Model(decoder_inputs, decoder_outputs, name="decoder")

  

@tf.keras.utils.register_keras_serializable()

class VQVAE(keras.Model):

    def __init__(self, train_variance, embedding_dim, num_embeddings, beta=0.25, rec_loss_weight=1.0, **kwargs):

        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim; self.num_embeddings = num_embeddings; self.beta = beta

        self.rec_loss_weight = rec_loss_weight; self.train_variance = train_variance

        self.encoder = build_vq_encoder(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), embedding_dim=self.embedding_dim)

        self.quantizer = VectorQuantizer(self.num_embeddings, self.embedding_dim, self.beta, name="vector_quantizer")

        decoder_input_shape = self.encoder.output_shape[1:]

        self.decoder = build_vq_decoder(decoder_input_shape, IMG_CHANNELS)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")

        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

        self.mse_loss_fn = tf.keras.losses.MeanSquaredError() # Use standard MSE

  

    @property

    def metrics(self): return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.vq_loss_tracker]

    def call(self, inputs): x = self.encoder(inputs); quantized = self.quantizer(x); return self.decoder(quantized)

    def train_step(self, data):

        if isinstance(data, tuple): data = data[0]

        with tf.GradientTape() as tape:

            encoder_outputs = self.encoder(data); quantized_latents = self.quantizer(encoder_outputs); reconstructions = self.decoder(quantized_latents)

            rec_loss = self.mse_loss_fn(data, reconstructions) / self.train_variance

            commitment_loss = sum(self.quantizer.losses)

            total_loss = self.rec_loss_weight * rec_loss + commitment_loss

        grads = tape.gradient(total_loss, self.trainable_variables); self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss); self.reconstruction_loss_tracker.update_state(rec_loss); self.vq_loss_tracker.update_state(commitment_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):

        if isinstance(data, tuple): data = data[0]

        encoder_outputs = self.encoder(data); quantized_latents = self.quantizer(encoder_outputs); reconstructions = self.decoder(quantized_latents)

        rec_loss = self.mse_loss_fn(data, reconstructions) / self.train_variance

        commitment_loss = sum(self.quantizer.losses)

        total_loss = self.rec_loss_weight * rec_loss + commitment_loss

        self.total_loss_tracker.update_state(total_loss); self.reconstruction_loss_tracker.update_state(rec_loss); self.vq_loss_tracker.update_state(commitment_loss)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self): config = super().get_config(); config.update({ "train_variance": float(self.train_variance.numpy()) if tf.is_tensor(self.train_variance) else float(self.train_variance), "embedding_dim": self.embedding_dim, "num_embeddings": self.num_embeddings, "beta": self.beta, "rec_loss_weight": self.rec_loss_weight }); return config

    @classmethod

    def from_config(cls, config): return cls(**config)

  

# --- Build and Compile VQ-VAE ---

print("\nBuilding VQ-VAE...")

if 'train_data' not in locals() or train_data is None or len(train_data) == 0: print("ERROR: train_data not available."); exit()

data_variance = tf.cast(tf.nn.moments(train_data, axes=[0, 1, 2, 3])[1], dtype=tf.float32)

print(f"Calculated training data variance: {data_variance:.4f}")

if data_variance < 1e-6: print("Warning: Low data variance. Setting to 1.0."); data_variance = 1.0

vqvae = VQVAE(train_variance=data_variance, embedding_dim=EMBEDDING_DIM, num_embeddings=NUM_EMBEDDINGS, beta=COMMITMENT_BETA, rec_loss_weight=REC_LOSS_WEIGHT, name="vq_vae")

build_input_shape = (None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

vqvae.build(build_input_shape)

print(f"VQ-VAE model explicitly built with input shape: {build_input_shape}")

print("\nEncoder Summary:"); vqvae.encoder.summary(line_length=100)

print("\nDecoder Summary:"); vqvae.decoder.summary(line_length=100)

print("\nCompiling VQ-VAE...")

vqvae.compile(optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-5))

  

# --- Callbacks ---

print("\nSetting up callbacks...")

checkpoint_filepath = CHECKPOINT_DIR + 'vqvae_epoch_{epoch:02d}-val_rec_loss_{val_reconstruction_loss:.4f}.weights.h5'

vqvae_checkpoint_callback = callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='val_reconstruction_loss', mode='min', save_best_only=True, verbose=1)

vqvae_early_stopping = callbacks.EarlyStopping(monitor='val_reconstruction_loss', patience=EARLY_STOPPING_PATIENCE, mode='min', verbose=1, restore_best_weights=True)

vqvae_reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_reconstruction_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)

vqvae_callbacks_list = [vqvae_early_stopping, vqvae_reduce_lr, vqvae_checkpoint_callback]

  

# --- Train VQ-VAE ---

print("\nStarting VQ-VAE training...")

history = vqvae.fit(

    train_dataset,

    epochs=EPOCHS,

    validation_data=val_dataset,

    callbacks=vqvae_callbacks_list,

    verbose=1

)

print("VQ-VAE Training finished.")

  

# --- Save Final Model ---

print(f"\nSaving final VQ-VAE model (best weights) to {MODEL_SAVE_PATH}...")

try:

    vqvae.save(MODEL_SAVE_PATH) # Save in native Keras format

    print("VQ-VAE model saved successfully.")

except Exception as e:

    print(f"Error saving full VQ-VAE model: {e}")

    print("Saving weights only as fallback...")

    try: vqvae.save_weights(MODEL_SAVE_PATH.replace('.keras', '.weights.h5')); print("VQ-VAE weights saved successfully.")

    except Exception as e2: print(f"Error saving VQ-VAE weights: {e2}")

  

# --- Evaluation and Visualization ---

print("\n--- Evaluating VQ-VAE ---")

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

plt.plot(history.history.get('total_loss', []), label='Train Total Loss'); plt.plot(history.history.get('val_total_loss', []), label='Val Total Loss')

plt.plot(history.history.get('reconstruction_loss', []), label='Train Rec Loss (MSE/Var)', linestyle='--'); plt.plot(history.history.get('val_reconstruction_loss', []), label='Val Rec Loss (MSE/Var)', linestyle='--')

plt.title('VQ-VAE Total & Reconstruction Loss'); plt.ylabel('Loss'); plt.legend(); plt.grid(True); plt.ylim(bottom=0)

plt.subplot(2, 1, 2)

plt.plot(history.history.get('vq_loss', []), label='Train VQ (Commitment) Loss'); plt.plot(history.history.get('val_vq_loss', []), label='Val VQ (Commitment) Loss')

plt.title('VQ-VAE Commitment Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True); plt.ylim(bottom=0)

plt.tight_layout(); plt.show()

  

if val_data is not None and len(val_data) > 0:

    n_show = min(8, len(val_data)); indices = np.random.choice(len(val_data), n_show, replace=False); original_images = val_data[indices]

    print("Generating reconstructed images..."); reconstructed_images = vqvae.predict(original_images, batch_size=BATCH_SIZE)

    print(f"\nShowing {n_show} original vs reconstructed images and NDVI..."); vis_bands_indices = VGG_BANDS_INDICES

    all_ndvi_orig = []; all_ndvi_recon = []

    plt.figure(figsize=(n_show * 2.5, 10))

    for i in range(n_show):

        # Original Image (RGB)

        ax = plt.subplot(4, n_show, i + 1)

        try:

            orig_vis = original_images[i][:, :, vis_bands_indices]; p1, p99 = np.percentile(orig_vis[orig_vis > 0], (1, 99)); p99 = max(p99, p1 + 1e-6); orig_vis = np.clip((np.clip(orig_vis, p1, p99) - p1) / (p99 - p1), 0.0, 1.0); plt.imshow(orig_vis)

        except Exception as img_e: print(f"Error processing original image {i} for display: {img_e}"); plt.imshow(np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)))

        # **** Corrected Title Logic ****

        if i == 0:

            plt.title("Original\nRGB Vis")

        else:

            plt.title("Original")

        plt.axis("off")

        # *****************************

  

        # Reconstructed Image (RGB)

        ax = plt.subplot(4, n_show, i + 1 + n_show)

        try:

            recon_vis = reconstructed_images[i][:, :, vis_bands_indices]; p1, p99 = np.percentile(recon_vis[recon_vis > 0], (1, 99)); p99 = max(p99, p1 + 1e-6); recon_vis = np.clip((np.clip(recon_vis, p1, p99) - p1) / (p99 - p1), 0.0, 1.0); plt.imshow(recon_vis)

        except Exception as img_e: print(f"Error processing reconstructed image {i} for display: {img_e}"); plt.imshow(np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)))

        # **** Corrected Title Logic ****

        if i == 0:

             plt.title("Reconstructed\nRGB Vis")

        else:

             plt.title("Reconstructed")

        plt.axis("off")

        # *****************************

  

        # Original NDVI

        ax = plt.subplot(4, n_show, i + 1 + 2 * n_show)

        im_orig_ndvi = None

        try:

            ndvi_orig = calculate_ndvi(original_images[i], NDVI_NIR_INDEX, NDVI_RED_INDEX); im_orig_ndvi = plt.imshow(ndvi_orig, cmap='RdYlGn', vmin=-1, vmax=1); all_ndvi_orig.append(ndvi_orig)

        except Exception as ndvi_e: print(f"Error calculating/plotting orig NDVI {i}: {ndvi_e}"); plt.imshow(np.zeros((IMG_HEIGHT,IMG_WIDTH)), cmap='gray');

        # **** Corrected Title Logic ****

        if i == 0:

             plt.title("Original\nNDVI")

             if im_orig_ndvi: plt.colorbar(im_orig_ndvi, ax=ax, fraction=0.046, pad=0.04)

        else:

             plt.title("Original")

        plt.axis("off")

        # *****************************

  

        # Reconstructed NDVI

        ax = plt.subplot(4, n_show, i + 1 + 3 * n_show)

        im_recon_ndvi = None

        try:

            ndvi_recon = calculate_ndvi(reconstructed_images[i], NDVI_NIR_INDEX, NDVI_RED_INDEX); im_recon_ndvi = plt.imshow(ndvi_recon, cmap='RdYlGn', vmin=-1, vmax=1); all_ndvi_recon.append(ndvi_recon)

        except Exception as ndvi_e: print(f"Error calculating/plotting recon NDVI {i}: {ndvi_e}"); plt.imshow(np.zeros((IMG_HEIGHT,IMG_WIDTH)), cmap='gray');

        # **** Corrected Title Logic ****

        if i == 0:

            plt.title("Reconstructed\nNDVI")

            if im_recon_ndvi: plt.colorbar(im_recon_ndvi, ax=ax, fraction=0.046, pad=0.04)

        else:

             plt.title("Reconstructed")

        plt.axis("off")

        # *****************************

  

    plt.tight_layout(); plt.show()

    if all_ndvi_orig and all_ndvi_recon and len(all_ndvi_orig) == len(all_ndvi_recon):

        all_ndvi_orig = np.array(all_ndvi_orig); all_ndvi_recon = np.array(all_ndvi_recon)

        ndvi_mae = np.mean(np.abs(all_ndvi_orig - all_ndvi_recon)); ndvi_mse = np.mean((all_ndvi_orig - all_ndvi_recon)**2)

        print(f"\nNDVI Comparison (for {n_show} visualized samples):"); print(f"  MAE: {ndvi_mae:.4f}, MSE: {ndvi_mse:.4f}")

    else: print("\nCould not perform quantitative NDVI comparison.")

else: print("Validation data not available for visualization.")

  

# End time

end_time = time.time()

print(f"\nScript finished in {end_time - start_time:.2f} seconds.")
