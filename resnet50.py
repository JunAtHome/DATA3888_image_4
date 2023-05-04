import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array



# Define your dataset directory
dataset_dir = 'C:/Users/Jun/Documents/DATA3888_image_4/data/data_processed/cell_images_cleaned'

class_names = os.listdir(dataset_dir)
class_indices = {name: i for i, name in enumerate(class_names)}

images = []
labels = []

for label, class_name in enumerate(class_names):
    class_directory = os.path.join(dataset_dir, class_name)
    for image_name in os.listdir(class_directory):
        images.append(os.path.join(class_directory, image_name))
        labels.append(label)

# Train validation split
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)

train_df = pd.DataFrame({'filename': train_images, 'class': train_labels})
val_df = pd.DataFrame({'filename': val_images, 'class': val_labels})

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=None,
    x_col='filename',
    y_col='class',
    target_size=(32, 32),
    batch_size=64,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_dataframe(
    val_df,
    directory=None,
    x_col='filename',
    y_col='class',
    target_size=(32, 32),
    batch_size=64,
    class_mode='categorical'
)



# # Set up data generators for training and validation
# train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# train_generator = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=(32, 32),
#     batch_size=64,
#     class_mode='categorical',
#     subset='training'
# )

# validation_generator = train_datagen.flow_from_directory(
#     dataset_dir,
#     target_size=(32, 32),
#     batch_size=64,
#     class_mode='categorical',
#     subset='validation'
# )

# Load the pre-trained ResNet50 model without the top (classification) layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# Fine-tune only the top layers, freeze the rest
for layer in base_model.layers:
    layer.trainable = False


# Add a new classification head to the base model
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
predictions = layers.Dense(train_generator.num_classes, activation='softmax')(x)
# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
model_history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=100
)
# Save the fine-tuned model
model.save("resnet_50_cells_100_epoch.h5")


results = []

for index, row in val_df.iterrows():
    image_path = row['filename']
    true_label = row['class']
    image_id = os.path.basename(image_path).split('.')[0]  # Assuming the ID is the filename without the extension

    img_array = preprocess_image(image_path, target_size=(32, 32))
    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions, axis=1)[0]

    is_correct = true_label == predicted_label
    results.append([image_id, is_correct])


results_df = pd.DataFrame(results, columns=['Image_ID', 'Prediction_Correct'])
results_df.to_csv('validation_results.csv', index=False)

with open('/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(model_history.history, file_pi)