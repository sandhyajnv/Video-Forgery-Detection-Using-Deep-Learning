
import os
import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping


model1=keras.models.load_model("/home/arnav/Downloads/Sandhya/videoforgeryCNNpart.h5")
intermediate_layer_model=keras.Model(inputs=model1.input,outputs=model1.layers[8].output)

# Define functions



    
def convert_to_3d(frames, T, overlap):
    num_frames = len(frames)
    height, width = frames[0].shape
    num_3d_frames = (num_frames - overlap) // (T - overlap)

    dta = []
    
    for i in range(num_3d_frames):
        result = np.stack(frames[i * (T - overlap) : i * (T - overlap) + T], axis=-1)
        dta.append(result)

    return dta,num_3d_frames 
    
    

    
    

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
    videolenpointer=[]
    print("processing video batch............")
    batched_data_3d = []
    
    for i in range(0, len(video_data_list), batch_size):
        batch = video_data_list[i : i + batch_size]
        
        for data in batch:
            data_frames = read_video_frames(data)
            d,l=convert_to_3d(data_frames, T, overlap)
            batched_data_3d.extend(d)
            videolenpointer.append(l)
            
            del data_frames
            
    return batched_data_3d,videolenpointer



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
authentic_data3d ,authvideolenpointer= process_video_batch(Authpath, T, overlap, batch_size)
tampered_data3d,tampvideolenpointer = process_video_batch(Forgedpath, T, overlap, batch_size)

Xauthentic_data3d =np.array(authentic_data3d )
Xtampered_data3d =np.array(tampered_data3d )

Xauth=intermediate_layer_model.predict(Xauthentic_data3d)
Xtamp=intermediate_layer_model.predict(Xtampered_data3d)


print("shape of new data",Xauth.shape)
print("shape of new data",Xauth.shape)



Xauth=list(Xauth)
Xtamp=list(Xtamp)


XauthCutVideodata=[]
XtampCutVideodata=[]

sum=0
for i in authvideolenpointer:
    XauthCutVideodata.append(Xauth[sum:sum+i])
    sum=sum+i
    
sum=0  
for i in tampvideolenpointer:
    XtampCutVideodata.append(Xtamp[sum:sum+i])
    sum=sum+i


max_len=0
for data in XauthCutVideodata:
  if len(data)>max_len:
    max_len=len(data)

for data in XtampCutVideodata:
  if len(data)>max_len:
    max_len=len(data)
    
print(len(XauthCutVideodata),len(XtampCutVideodata )) 
print("max_length",max_len)

for data in XauthCutVideodata:
  if len(data)<max_len:
    for i in range(max_len-len(data)):
      data.append(np.zeros(len(data[0])))


for data in XtampCutVideodata:
  if len(data)<max_len:
    for i in range(max_len-len(data)):
      data.append(np.zeros(len(data[0])))





# Convert lists to NumPy arrays for better performance
Xauthentic = np.array(XauthCutVideodata)
Xtampered = np.array(XtampCutVideodata)



# Concatenate Xauthentic and Xtampered arrays
X = np.concatenate((Xauthentic, Xtampered), axis=0)

print("final data shape",X.shape)
# Create labels Y
Y = np.concatenate((np.zeros(len(Xauthentic)), np.ones(len(Xtampered))), axis=0)
print("final label shape",Y.shape)

# Split data into train, validation, and test sets
from sklearn.model_selection import train_test_split
X_Train, X_Val, Y_Train, Y_Val = train_test_split(X, Y , test_size=0.2, stratify=Y)
# Split data into train, validation, and test sets
#total_samples = X.shape[0]
#train_size = int(total_samples * 0.8)
#val_size = int(total_samples * 0.2)

#X_Train = X[:train_size]
#X_Val = X[train_size : ]


#Y_Train = Y[:train_size]
#Y_Val = Y[train_size : ]


#Y_Train = to_categorical(Y_Train-1, num_classes=2)
#Y_Val = to_categorical(Y_Val-1, num_classes=2)


print("X_Train shape:", X_Train.shape)
print("Y_Train shape:", Y_Train.shape)
print("X_Val shape:", X_Val.shape)
print("Y_Val shape:", Y_Val.shape)


from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout,Masking





# Define the LSTM model
model = keras.Sequential([
    Masking(mask_value=0.0,input_shape=(None, 128)),
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(None,128)),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dropout(0.5),  # Add dropout for regularization
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # Output layer with 2 classes ('forged' and 'unforged')
])

model.summary()
# Create the Adam optimizer without the 'momentum' argument
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
# Compile the model
model.compile(optimizer=optimizer ,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_Train, Y_Train, epochs=200, batch_size=10, validation_data=(X_Val, Y_Val),callbacks=[early_stopping],shuffle=True)


model.save('videoforgeryLSTMpart.h5')

# # Evaluate the model on the test set
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')
