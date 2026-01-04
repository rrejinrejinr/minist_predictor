import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")
st.title("🧠 MNIST Digit Classification")
st.write("CNN model using TensorFlow & Streamlit")

@st.cache_resource
def load_data_and_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0)
    return model, x_test, y_test

model, x_test, y_test = load_data_and_model()

st.subheader("📊 Model Accuracy")
loss, acc = model.evaluate(x_test, y_test, verbose=0)
st.success(f"Accuracy: {acc * 100:.2f}%")

st.subheader("🔢 Predict a Random Digit")

if st.button("Predict"):
    idx = np.random.randint(0, len(x_test))
    img = x_test[idx]
    true_label = y_test[idx]

    pred = model.predict(img.reshape(1,28,28,1))
    predicted_label = np.argmax(pred)

    fig, ax = plt.subplots()
    ax.imshow(img.reshape(28,28), cmap="gray")
    ax.axis("off")
    st.pyplot(fig)

    st.write("✅ **Predicted Digit:**", predicted_label)
    st.write("🎯 **Actual Digit:**", true_label)

    if predicted_label == true_label:
        st.success("Correct Prediction 🎉")
    else:
        st.error("Wrong Prediction ❌")
