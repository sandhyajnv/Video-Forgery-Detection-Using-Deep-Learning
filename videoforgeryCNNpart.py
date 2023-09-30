import os
import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# Define functions



    
def convert_to_3d(frames, T, overlap):
    num_frames = len(frames)
    height, width = frames[0].shape
    num_3d_frames = (num_frames - overlap) // (T - overlap)

    dta = []
    
    for i in range(num_3d_frames):
        result = np.stack(frames[i * (T - overlap) : i * (T - overlap) + T], axis=-1)
        dta.append(result)

    return dta
    
    

    
    

def read_video_frames(video_path, num_frames=None):
    frames = []
    vid = cv2.VideoCapture(video_path)
    
    ret = True
    frame_count = 0
    while ret and (num_frames is None or frame_count < num_frames):
        ret, frame = vid.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.resize(gray_frame, (480, 720))
            frames.append(gray_frame)
            frame_count += 1
    
    vid.release()  # Release the video capture object
    return frames








def process_video_batch(video_data_list, T, overlap, batch_size):
    print("processing video batch............")
    batched_data_3d = []
    
    for i in range(0, len(video_data_list), batch_size):
        batch = video_data_list[i : i + batch_size]
        
        for data in batch:
            data_frames = read_video_frames(data)
            batched_data_3d.extend(convert_to_3d(data_frames, T, overlap))
            
            del data_frames
            
    return batched_data_3d



# Load video paths
video_paths = []
for dirname, _, filenames in os.walk('/media/arnav/MEDIA/SandhyaIntraframe'):
    for filename in filenames:
        video_paths.append(os.path.join(dirname, filename))

# Divide paths into Authentic and Forged
Authpath = [path for path in video_paths if path.endswith('.mp4') and path.split("_")[1][:4] == "Auth"]
Forgedpath = [path for path in video_paths if path.endswith('.mp4') and path.split("_")[1][:4] != "Auth"]
Authpath = Authpath
Forgedpath = Forgedpath
print("length of forged path")
print(len(Forgedpath))
# Process video batches
batch_size = 1
T = 2
overlap = 1
authentic_data3d = process_video_batch(Authpath, T, overlap, batch_size)
tampered_data3d = process_video_batch(Forgedpath, T, overlap, batch_size)

# Convert lists to NumPy arrays for better performance
Xauthentic = np.array(authentic_data3d)
Xtampered = np.array(tampered_data3d)

# Concatenate Xauthentic and Xtampered arrays
X = np.concatenate((Xauthentic, Xtampered), axis=0)

# Create labels Y
Y = np.concatenate((np.ones(len(Xauthentic)), 2 * np.ones(len(Xtampered))), axis=0)
print(Y.shape)

# Split data into train, validation, and test sets

# Split data into train, validation, and test sets

from sklearn.model_selection import train_test_split
X_Train, X_Val, Y_Train, Y_Val = train_test_split(X, Y , test_size=0.2, stratify=Y)

#total_samples = X.shape[0]
#train_size = int(total_samples * 0.8)
#val_size = int(total_samples * 0.2)

#X_Train = X[:train_size]
#X_Val = X[train_size : ]


#Y_Train = Y[:train_size]
#Y_Val = Y[train_size : ]


Y_Train = to_categorical(Y_Train-1, num_classes=2)
Y_Val = to_categorical(Y_Val-1, num_classes=2)


# Create the 2D convolutional model
#@keras.saving.register_keras_serializable(package="my_package", name="generate_custom_kernel")
def generate_custom_kernel(T, M, N):

    a = []
    kernal = []

    for i in range(T):
        if i == T // 2:
            a.append((T - 1) / T)
        else:
            a.append(1 / T)

    for i in range(M):
        c = []
        for j in range(N):
            c.append(a)
        kernal.append(c)

    return np.array(kernal)

#custom kernel initializer
T = 2
M = 3
N = 3
kernal = generate_custom_kernel(T, M, N)

#, kernel_initializer=tf.constant_initializer(kernal)

def create_2d_conv_model(input_shape, kernel):
    model = keras.Sequential([
        keras.layers.Conv2D(1, (3, 3), padding='same', input_shape=input_shape),
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(2, activation="softmax")
    ])
    return model

input_shape = (720, 480, 2)
model = create_2d_conv_model(input_shape, kernal)
intermediate_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[8].output)

model.summary()





# Compile the model
initial_learning_rate = 0.001
decay_rate = 0.9
decay_step = 5000
optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate, momentum=0.9)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define learning rate schedule function
def lr_schedule(epoch, lr):
    return lr * tf.math.pow(decay_rate, tf.math.floor((epoch + 1) / decay_step))

# callback for learning rate scheduling
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)


checkpoint_filepath = 'model_checkpoint.h5'
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)





early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)



# Train the model
batch_size = 20
max_epochs = 200


# Train the model with early stopping
history = model.fit(
    X_Train, Y_Train,
    batch_size=batch_size,
    epochs=max_epochs,
    validation_data=(X_Val, Y_Val),
    callbacks=[lr_scheduler, model_checkpoint, early_stopping]
)



model.save('videoforgeryCNNpart.h5')



