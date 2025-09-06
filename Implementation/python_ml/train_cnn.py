import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, concatenate
from keras.models import Model
import sys
import joblib

if len(sys.argv) > 1:
    accumulated_data = joblib.load(sys.argv[1]) 
    df = pd.DataFrame(accumulated_data)
else:
    df = pd.read_csv('synthetic_data.csv')

X_vib = np.array(df['vibration'].apply(lambda x: eval(x) if isinstance(x, str) else x).tolist())
X_vib = X_vib.reshape((X_vib.shape[0], X_vib.shape[1], 1))
X_temp = df['temp'].values.reshape(-1, 1)
X_volt = df['voltage'].values.reshape(-1, 1)
y = df['label'].values

vib_input = Input(shape=(100, 1))
conv = Conv1D(64, kernel_size=3, activation='relu')(vib_input)
pool = MaxPooling1D(2)(conv)
flat = Flatten()(pool)
temp_input = Input(shape=(1,))
volt_input = Input(shape=(1,))
merged = concatenate([flat, temp_input, volt_input])
dense = Dense(128, activation='relu')(merged)
output = Dense(1, activation='sigmoid')(dense)
model = Model(inputs=[vib_input, temp_input, volt_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit([X_vib, X_temp, X_volt], y, epochs=10, batch_size=32, validation_split=0.2)
model.save('model.h5')
print("Trained and saved model.h5")