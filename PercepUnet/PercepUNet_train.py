import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import os


# build model
def build_unet(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # decoder
    u2 = layers.UpSampling2D((2, 2))(c3)
    u2 = layers.concatenate([u2, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u1 = layers.UpSampling2D((2, 2))(c4)
    u1 = layers.concatenate([u1, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(c5)
    return tf.keras.Model(inputs, outputs)


# feature extractor, here we use VGG16, you can use your own one.
def build_feature_extractor():
    vgg = VGG16(weights="imagenet", include_top=False)
    layer_names = ['block1_conv2', 'block2_conv2']
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = Model(inputs=vgg.input, outputs=outputs)
    model.trainable = False
    return model



def feature_loss(noisy_image, denoised_image, feature_extractor):
    noisy_features = feature_extractor(noisy_image)
    denoised_features = feature_extractor(denoised_image)

    loss = 0
    for nf, df in zip(noisy_features, denoised_features):
        loss += tf.reduce_mean(tf.abs(nf - df))  # L1 norm
    return loss


# loss function
def total_loss(noisy_image, denoised_image, feature_extractor, alpha=0.5):
    f_loss = feature_loss(noisy_image, denoised_image, feature_extractor)
    pixel_loss = tf.reduce_mean(tf.square(denoised_image - noisy_image))
    return alpha * f_loss + (1 - alpha) * pixel_loss


# prepare data
def prepare_dataset_from_folder(folder_path, image_size=(128, 128), batch_size=8):

    dataset = tf.keras.utils.image_dataset_from_directory(
        folder_path,
        label_mode=None,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )
    # normalize to [0, 1]
    dataset = dataset.map(lambda x: x / 255.0)
    return dataset


def train_denoising_model(dataset, model, feature_extractor, optimizer, batch_size=8, epochs=20, alpha=0.7):
    steps_per_epoch = len(dataset)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0


        for step, noisy_images in enumerate(dataset):
            with tf.GradientTape() as tape:
                denoised_images = model(noisy_images, training=True)
                loss = total_loss(noisy_images, denoised_images, feature_extractor, alpha)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))


            epoch_loss += loss.numpy()


        average_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch + 1}, Average Loss: {average_loss:.4f}")

    # save model
    model_save_path = "denoising_model_all_nosie.h5"
    model.save(model_save_path)
    print(f"\nModel saved to {model_save_path}")



if __name__ == "__main__":
    # train data
    folder_path = "augmented_images"

    image_size = (64, 64)
    batch_size = 8
    dataset = prepare_dataset_from_folder(folder_path, image_size=image_size, batch_size=batch_size)

    input_shape = image_size + (3,)
    model = build_unet(input_shape)
    feature_extractor = build_feature_extractor()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    train_denoising_model(dataset, model, feature_extractor, optimizer, batch_size=batch_size, epochs=30, alpha=0.7)
