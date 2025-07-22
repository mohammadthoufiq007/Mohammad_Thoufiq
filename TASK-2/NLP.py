import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Define constants
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 200

def load_and_preprocess_data():
    """Loads and preprocesses the IMDB dataset"""
    print("Loading data...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
    X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
    X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
    return X_train, y_train, X_test, y_test

def create_model():
    """Creates a CNN-LSTM model for sentiment analysis"""
    model = keras.Sequential([
        layers.Embedding(VOCAB_SIZE, 128, input_length=MAX_SEQUENCE_LENGTH),
        layers.Conv1D(32, 5, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def train_model(model, X_train, y_train):
    """Trains the model on the training data"""
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.2
    )
    return history

def plot_training_results(history):
    """Plots the training and validation loss and accuracy"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test data"""
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")

def make_prediction(model, X_test, y_test, sample_index):
    """Makes a prediction on a sample review"""
    sample_review = X_test[sample_index].reshape(1, -1)
    prediction = model.predict(sample_review)

    print("\nSample Review Prediction:")
    print("Predicted Sentiment:", "Positive" if prediction[0][0] > 0.5 else "Negative")
    print("Actual Sentiment:", "Positive" if y_test[sample_index] == 1 else "Negative")

def main():
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    model = create_model()
    model.summary()
    history = train_model(model, X_train, y_train)
    plot_training_results(history)
    evaluate_model(model, X_test, y_test)
    make_prediction(model, X_test, y_test, 123)

if __name__ == "__main__":
    main()
