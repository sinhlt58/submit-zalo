import argparse
import csv
import collections

import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer

from glue import glue_convert_examples_to_features
from utils import read_json_data, write_json_data


class DataProcessor:

    def __init__(self):
        self.char_to_remove_in_question = b',"-\xe2\x80\x9c\xe2\x80\x9d()?\xe2\x80\x93/+\x08!:._\xc2\xad\'\xe2\x80\xb2\\\xe2\x80\x98\xe2\x80\x99'.decode('utf-8')

        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        print ("Done init bert_tokenizer!")

    def convert_zalo_to_tfrecord(self, data_dir, file_name):
        if file_name not in ["train", "test", "private"]:
            print ("Invalid file_name {}".format(file_name))
            return

        self._zalo_to_flat_json(data_dir, file_name)

    def _int64_feature(self, values):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

    def _serialize_example(self, features_dict, label):
        feature = collections.OrderedDict()
        feature["input_ids"] = self._int64_feature(features_dict["input_ids"].numpy().tolist())
        feature["attention_mask"] = self._int64_feature(features_dict["attention_mask"].numpy().tolist())
        feature["token_type_ids"] = self._int64_feature(features_dict["token_type_ids"].numpy().tolist())
        feature["label"] = self._int64_feature([int(label.numpy())])

        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
        return tf_example.SerializeToString()

    def _zalo_to_flat_json(self, data_dir, file_name):
        in_json_path = "{}/{}.json".format(data_dir, file_name)
        out_tsv_path = "{}/{}.tsv".format(data_dir, file_name)
        out_pids_path = "{}/{}_pids.txt".format(data_dir, file_name)

        data_json = read_json_data(in_json_path)
        # convert to flat converted examples
        i = 0
        converted_samples = []
        for sample_json in data_json:
            converted_sample = None

            if file_name in ["train", "squad"]:
                converted_sample = sample_json
                converted_sample["pid"] = "p1"
                converted_samples.append(converted_sample)

            elif file_name in ["test", "private"]:
                for p in sample_json["paragraphs"]:
                    if 'label' in p:
                        label = True if p['label'] == '1' else False
                    else:
                        label = False
                    converted_sample = {
                        "id": sample_json["__id__"],
                        "title": sample_json["title"],
                        "question": self._pre_process_question(sample_json["question"]),
                        "text": self._pre_process_text(p["text"]),
                        "label": label,
                        "pid": p["id"]
                    }
                    converted_samples.append(converted_sample)

        # convert to tsv format
        tsv_dict = {
            "index": [],
            "question": [],
            "sentence": [],
            "label": [],
        }
        id_pids = []
        for idx, json_sample in enumerate(converted_samples):
            tsv_dict["index"].append(idx)
            tsv_dict["question"].append(json_sample["question"])
            tsv_dict["sentence"].append(json_sample["text"])
            tsv_dict["label"].append(
                "entailment" if json_sample["label"] else "not_entailment"
            )
            id_pid = "{}@{}".format(json_sample["id"], json_sample["pid"])
            id_pids.append(id_pid)

        tsv_df = pd.DataFrame(tsv_dict)

        self._write_tsv(out_tsv_path, tsv_df)
        self._write_pids(out_pids_path, id_pids)

        # convert to tfrecord format
        tsv_df["idx"] = tsv_df["index"]
        tf_dataset = tf.data.Dataset.from_tensor_slices(dict(tsv_df))

        features_dataset = glue_convert_examples_to_features(tf_dataset, self.bert_tokenizer, 512, 'qnli')

        def gen():
            for features_dict, label in features_dataset:
                yield self._serialize_example(features_dict, label)

        serialized_features_dataset = tf.data.Dataset.from_generator(
            gen, output_types=tf.string, output_shapes=())

        num_examples = tsv_df.shape[0]
        print ("num_examples test: ", num_examples)
        record_path = "{}/{}.tfrecord".format(
            data_dir, file_name
        )
        writer = tf.data.experimental.TFRecordWriter(record_path)
        writer.write(serialized_features_dataset)
        print ("Write file {}".format(record_path))

    def _pre_process_common(self, text):
        tokens = text.split()
        text = " ".join(tokens)
        return text.strip()

    def _pre_process_question(self, text):
        text = self._pre_process_common(text)

        # 1. remove special characters in questions
        r_text = ""
        for c in text:
            if c in self.char_to_remove_in_question:
                r_text += ' ' # we replace them with ' '
            else:
                r_text += c
        text = self._pre_process_common(r_text)

        # 2. Uppercase the first character
        text = text[:1].capitalize() + text[1:]

        # 3. We add ? to the end of the question
        text += " ?"

        return text

    def _pre_process_text(self, text):
        text = self._pre_process_common(text)

        return text

    def _write_tsv(self, tsv_file, df):
        df.to_csv(tsv_file, encoding="utf-8", quoting=csv.QUOTE_NONE, sep="\t", index=False,
                    columns=["index", "question", "sentence", "label"])
        print ("Write file {}".format(tsv_file))

    def _write_pids(self, pids_file, id_pids):
        with open(pids_file, "w") as f:
            for pid in id_pids:
                f.write("{}\n".format(pid))
        print ("Write file {}".format(pids_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data", required=True, type=str,
                        help="The data folder")
    parser.add_argument("--file_name", default="test", required=True, type=str,
                        help="The input data file is a zalo format .json file."
                              "It takes only one of train|test|private")

    args = parser.parse_args()

    data_processor = DataProcessor()
    data_processor.convert_zalo_to_tfrecord(args.data_dir, args.file_name)
