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
  && pip install --upgrade jax jaxlib \
  && pip install pandas \
  && pip install hdf5storage \
  && pip install h5py

RUN jupyter notebook --generate-config

RUN echo "" > ~/.jupyter/jupyter_notebook_config.py  \
	&& echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py \ 
	&& echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py \
	&& echo "c.NotebookApp.allow_root=True" >> ~/.jupyter/jupyter_notebook_config.py \
	&& echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py \
	&& echo "c.NotebookApp.notebook_dir = '/ws'" >> ~/.jupyter/jupyter_notebook_config.py \
	&& echo "c.NotebookApp.password = u'sha1:eb8c10bebe31:c8689687aab1e53c5cf0d6c4ce7620c6a6187e7b'" >> ~/.jupyter/jupyter_notebook_config.py

