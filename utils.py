import io
import os
import json
import pickle
import csv
import re

import tensorflow as tf
import numpy as np

def write_json_data(path, data):
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def read_json_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)

def dump_object(obj, path):
    pickle.dump(obj, open(path, 'wb'))

def get_method_key(key, method):
        return "{}_{}".format(method, key)

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_dir_names(path):
    return next(os.walk(path))[1]

bert_models_map = {
    "caseds": "cased_L-12_H-768_A-12",
    "casedl": "cased_L-24_H-1024_A-16",
    "uncaseds": "uncased_L-12_H-768_A-12",
    "uncasedl": "uncased_L-24_H-1024_A-16",
}

def get_bert_paths(bert_type):
    model_folder = bert_models_map[bert_type]
    model_path = "D:/works/zalo_challenges/models/bert/{}".format(model_folder)
    vocab_file = "{}/vocab.txt".format(model_path)

    do_lower_case = bert_type[:2] == "un"

    return model_path, vocab_file, do_lower_case

def convert_df_str(df):
    str_df = df.select_dtypes([np.object])
    str_df = str_df.stack().str.decode('utf-8').unstack()

    for col in str_df:
        df[col] = str_df[col]


def _load_glue_dataset(file_path):
    dict_data = {
        "idx": [],
        "label": [],
        "question": [],
        "sentence": [],
    }
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for idx, row in enumerate(reader):
            dict_data["idx"].append(idx)
            dict_data["label"].append(row[2])
            dict_data["question"].append(row[0])
            dict_data["sentence"].append(row[1])
    tf_dataset = tf.data.Dataset.from_tensor_slices(dict_data)
    return tf_dataset, len(dict_data["label"])

def load_glue_data(folder_path, task="qnli"):
    train_set, train_len = _load_glue_dataset("{}/{}/train.tsv".format(folder_path, task))
    dev_set, dev_len = _load_glue_dataset("{}/{}/dev.tsv".format(folder_path, task))

    return (train_set, train_len), (dev_set, dev_len)


def get_file_names_in_dir(dir_path):
    from os import walk

    f = []
    for (dirpath, dirnames, filenames) in walk(dir_path):
        f.extend(filenames)
        names = []
        for n in f:
            if n.endswith(".txt"):
                n = n.replace(".txt", "")
                names.append(n)
        return names

def find_dict_data(value, dict_data, key="id"):
    for sample in dict_data:
        if sample[key] == value:
            return sample
    return None

def process_split_text(text):
    tokens = text.split()
    text = " ".join(tokens)
    return text.strip()

def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()

def _to_utf8(file_path, out_path):
    json_dict = read_json_data(file_path)
    write_json_data(out_path, json_dict)

if __name__ == "__main__":
    _to_utf8("./bert-vietnamese-question-answering/dataset/dev-v2.0.json", "squad_data/vi_g_dev-v2.0.json")
    _to_utf8("./bert-vietnamese-question-answering/dataset/train-v2.0.json", "squad_data/vi_g_train-v2.0.json")
