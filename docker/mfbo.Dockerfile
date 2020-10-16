FROM tensorflow/tensorflow:1.15.2-gpu-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && apt-get install python3-tk -y \
  && pip3 install --upgrade pip\
  && pip install jupyter \
  && pip install scipy \
  && pip install matplotlib \
  && pip install tqdm \
  && pip install -U scikit-learn \
  && pip install tensorflow_probability==0.8 \
  && pip install hdf5storage \
  && pip install h5py


