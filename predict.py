import datetime
import json
import os
import glob
import pprint
import random
import string
import sys
import numpy as np
import scipy
import pandas as pd
import argparse
import tensorflow as tf
from transformers import *


def get_dataset(record_path, max_seq_len=512, class_weights=[]):
    """
      Because distributed strategy does not support class_weights yet.
      so we will convert class weights to sample weights instead.
    """
    name_to_features = {
      "input_ids": tf.io.FixedLenFeature([max_seq_len], tf.int64),
      "attention_mask": tf.io.FixedLenFeature([max_seq_len], tf.int64),
      "token_type_ids": tf.io.FixedLenFeature([max_seq_len], tf.int64),
      "label": tf.io.FixedLenFeature([], tf.int64),
    }

    def _decode_record(example_proto):
        """Decodes a record to a TensorFlow example."""
        example_dict = tf.io.parse_single_example(example_proto, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        features_dict = {}
        for name in list(example_dict.keys()):
            t = example_dict[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            if name == "label" and t.dtype != tf.int32:
                t = tf.cast(t, tf.int32)
                print (t)

            example_dict[name] = t
            if name != "label":
                features_dict[name] = t

        label = example_dict["label"]
        
        if class_weights:
            class_weight = tf.constant(1, tf.float32)
            for i, w in enumerate(class_weights):
                if i == label:
                    class_weight = tf.constant(class_weights[i])
            return (features_dict, label, class_weight)
        else:
            return (features_dict, label)

    filenames = [record_path]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    print ("raw_dataset: ", raw_dataset)
    parsed_dataset = raw_dataset.map(_decode_record)

    return parsed_dataset

def get_model(lr, eps):
    opt = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=eps, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
    # model = TFXLMForSequenceClassification.from_pretrained('xlm-mlm-17-1280')
    # model = TFXLMForSequenceClassification.from_pretrained('xlm-mlm-xnli15-1024')
    model.compile(optimizer=opt, loss=loss, metrics=[metric])

    return model

def ensemble_predictions(list_probs):
    arr = np.array(list_probs)
    combined_probs = np.mean(list_probs, axis=0)
    return combined_probs

def predict(data_dir, model_dir, out_dir, file_name):
    test_path_record = "{}/{}.tfrecord".format(data_dir, file_name)
    test_pids_path = "{}/{}_pids.txt".format(data_dir, file_name)
    best_models_path = "{}/best_models.json".format(model_dir)

    # get the test_dataset
    test_dataset = get_dataset(test_path_record, max_seq_len=512)
    test_dataset = test_dataset.batch(32, drop_remainder=False)
    print ("test_dataset", test_dataset)
    
    # get the best_models file
    with tf.io.gfile.GFile(best_models_path, 'r') as f:
        best_models = json.load(f)["best_models"]

    print ("best_models: ", best_models)
    num_best_models = sum(len(epochs) for _, epochs in best_models)
    print ("num_best_models: ", num_best_models)

    # get test_pids.txt file
    with tf.io.gfile.GFile(test_pids_path, 'r') as f:
        test_pids = [l.strip() for l in f.readlines()]
        test_num_pids = len(test_pids)
    print ("test_num_pids: ", test_num_pids)

    model = get_model(2e-5, 1e-7)

    # predict for each model and average their results
    do100 = True # for the full dataset
    do90 = False # for local train dataset split 90%
    list_probs = [] # for ensemble result later

    i = 0
    for model_name, ep_ckpts in best_models:
        model_data_type = model_name.split('_')[-1]
        if not do100 and model_data_type.startswith('100'):
            print ('Skip model 100: ', model_name)
            continue
        if not do90 and not model_data_type.startswith('100'):
            print ('Skip model 90: ', model_name)
            continue
        
        for ep_ckpt in ep_ckpts:
            # setup input model file, output file
            ckpt_path = "{}/{}/cp-{:02d}.ckpt".format(
                model_dir, model_name, ep_ckpt
            )
            test_result_path = "{}/{}_cp-{:02d}.json".format(
                out_dir, model_name, ep_ckpt
            )
            print ("predicting with model {} epoch {}".format(model_name, ep_ckpt))
            print ("ckpt_path: ", ckpt_path)
            print ("test_result_path: ", test_result_path)

            # load the new weights
            print ("loading the weights ...")
            model.load_weights("{}".format(ckpt_path))
            print ("start predicting ...")
            
            # write to file
            test_pred_logits = model.predict(test_dataset, verbose=1)
            test_pred_logits = test_pred_logits.tolist()

            test_logits = test_pred_logits[:test_num_pids] # cut off the padded samples
            test_probs = scipy.special.softmax(test_logits, axis=1).tolist()
            list_probs.append(test_probs)

            print ("test_logits len: ", len(test_logits))   
            with tf.io.gfile.GFile(test_result_path, "w") as f:
                json.dump({
                    "test_logits": test_logits,
                }, f)
            print ("Write file {}".format(test_result_path))

            i += 1
    
    # ensemble model results
    choose_probs = list_probs[:]
    print ("choose_probs len: ", len(choose_probs))
    combined_probs = ensemble_predictions(choose_probs)
    ensemble_preds = np.argmax(combined_probs, axis=1)
    print ("predict number has_answer: ", (ensemble_preds == 0).sum())

    # generate submission
    test_pids = np.array(test_pids)
    submit_id_pids = test_pids[ensemble_preds == 0]
    submit_test_ids = []
    submit_test_pids = []

    for id_pid in submit_id_pids:
        parts = id_pid.split('@')
        submit_test_ids.append(parts[0])
        submit_test_pids.append(parts[1])
        
    # generate the submission csv file
    submit_path = "{}/submission.csv".format(out_dir)
    submit_dict = {
        'test_id': submit_test_ids,
        'answer': submit_test_pids,
    }
    submit_df = pd.DataFrame(submit_dict)

    with tf.io.gfile.GFile(submit_path, 'w') as f:
        submit_df.to_csv(f, index=False)
        print ("Write file {}".format(submit_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data", required=True, type=str,
                        help="The data folder")
    parser.add_argument("--model_dir", default="saved_models", required=True, type=str,
                        help="The model dir")
    parser.add_argument("--out_dir", default="result", required=True, type=str,
                        help="The out dir")
    parser.add_argument("--file_name", default="test", required=True, type=str,
                        help="The file name without the .tfrecord extension")


    args = parser.parse_args()


    predict(args.data_dir, args.model_dir, args.out_dir, args.file_name)