import warnings

import joblib
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from utils import get_name, get_surrounding, get_token_tags, row2vec

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

FILE_PATH = "annotated data/"
W2V_PATH = "model/word2vec.model"
NEW_W2V_PATH = "model/word2vec_new.model"
LABEL2CLASS_PATH = "model/label2class.joblib"
MODEL_PATH = "model/trained_ann_model.h5"


def main() -> None:
    """
    Training ANN model.
    """
    num_classes = 8

    # load data and word2vec
    data_tags, data_tokens = get_token_tags(FILE_PATH)
    model = Word2Vec.load(W2V_PATH)

    # pre-process data
    data_feature = list(get_surrounding(x, 3) for x in data_tokens)

    # update vocab and retrain model
    model.build_vocab(data_tokens, update=True)
    model.train(data_tokens, total_examples=len(data_tokens), epochs=10)
    model.save(NEW_W2V_PATH)

    # similar word
    print(f"vocab size : {len(model.wv.key_to_index)}")

    # split data into train and test
    data_feature = [item for sublist in data_feature for item in sublist]
    label = [item for sublist in data_tags for item in sublist]
    print(f"data len {len(data_feature)}, label len {len(label)}")

    X_train, X_test, y_train, y_test = train_test_split(
        data_feature, label, test_size=0.2, random_state=42, stratify=label
    )
    print(f"train data len {len(X_train)}, test data len {len(X_test)}")

    # feature words into vector
    X_train = list(map(lambda x: row2vec(x, model), X_train))
    X_test = list(map(lambda x: row2vec(x, model), X_test))

    train_X = np.array(X_train).reshape(len(X_train), -1)
    test_X = np.array(X_test).reshape(len(X_test), -1)

    # Load label2class
    label2class = joblib.load(LABEL2CLASS_PATH)
    class2label = {value: key for key, value in label2class.items()}

    train_y = [class2label[tag] for tag in y_train]
    test_y = [class2label[tag] for tag in y_test]

    # Define the model
    ann_model = Sequential()
    ann_model.add(Dense(256, input_dim=train_X.shape[1], activation="relu"))
    ann_model.add(BatchNormalization())
    ann_model.add(Dropout(0.5))
    ann_model.add(Dense(128, activation="relu"))
    ann_model.add(BatchNormalization())
    ann_model.add(Dropout(0.5))
    ann_model.add(Dense(num_classes, activation="softmax"))

    # Compile the model
    ann_model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    # Define an EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True, verbose=False
    )

    # Train the model with EarlyStopping callback
    _ = ann_model.fit(
        train_X,
        np.array(train_y),
        epochs=200,
        batch_size=64,
        validation_split=0.15,
        callbacks=[early_stopping],
    )

    ann_model.save(MODEL_PATH)
    # ann_model.load_weights(MODEL_PATH)

    # Generate a classification report
    y_pred = ann_model.predict(test_X)
    y_pred_classes = tf.argmax(y_pred, axis=1)
    report = classification_report(
        test_y, y_pred_classes, target_names=label2class.values()
    )
    print(report)

    names, tags = get_name(
        "আব্দুর রহিম নামের কাস্টমারকে একশ টাকা বাকি দিলাম", model, ann_model, label2class
    )
    names, tags = get_name(
        "যুবলীগের কর্মী মাসুম ছাত্রদলের কর্মী সাদ্দামকে গুলি করেন",
        model,
        ann_model,
        label2class,
    )
    names, tags = get_name(
        "বন্দর থানার ভারপ্রাপ্ত কর্মকর্তা সঞ্জয় সিনহা বলেন", model, ann_model, label2class
    )
    print("training done!")


if __name__ == "__main__":
    main()
