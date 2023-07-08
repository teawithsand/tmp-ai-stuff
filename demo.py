import tensorflow as tf
import json
import keras


def sanitize(x):
    x = x.replace("\r", " ")
    x = x.replace("\n", " ")
    x = x.replace(" ,", ",")
    x = x.replace(" .", ".")
    for r in "[]{}()":
        x = x.replace(r, "")
    x = x.replace(" n't", "n't")
    x = x.replace(" 's", "'s")
    return x
maxlen = 35498

loaded = tf.keras.models.load_model("./trained-model-3.keras")
data = open("data.txt", "rt").read()
data = sanitize(data)


info = json.loads(open("tokenizer_config.json", "rt").read())
info = json.dumps({"config": info})
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(info)
data = tokenizer.texts_to_sequences([data])
data = keras.preprocessing.sequence.pad_sequences(data, maxlen=maxlen)
val = loaded.predict(data)[0]
print("")
if val <= 0:
    print("AI-generated")
else:
    print("Human generated")
print("Score", val)