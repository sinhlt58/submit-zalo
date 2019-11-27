# submit-zalo

### Train models.

Open the train.ipynb file with google colab.

We use tensorflow 2.0 and TPU to train our models.

You also need a google cloud storage bucket in order to train with TPU.

Change the ``BUCKET`` varabile to your bucket name. (Remember to update the bucket permission for the TPU serivce).

Upload the tfrerocd files in the ``/model/train_data/`` inside the docker container to the bucket.

Config the variables that are realated to the bucket path.
