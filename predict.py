import sys

import joblib
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model

from utils import get_name

NEW_W2V_PATH = "model/word2vec_new.model"
LABEL2CLASS_PATH = "model/label2class.joblib"
MODEL_PATH = "model/trained_ann_model2.h5"


def predict(sentence: str) -> None:
    """
    Get name from sentence
    """
    # Load word2vec, model & label2class
    model = Word2Vec.load(NEW_W2V_PATH)
    label2class = joblib.load(LABEL2CLASS_PATH)
    ann_model = load_model(MODEL_PATH)
    get_name(sentence, model, ann_model, label2class)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        predict(arg)
    else:
        print(
            "No arg!\nTry this example: python predict.py 'বন্দর থানার ভারপ্রাপ্ত কর্মকর্তা সঞ্জয় সিনহা বলেন'")
