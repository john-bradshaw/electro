FROM ubuntu:16.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion unzip gcc-5 g++-5

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh \
 && /bin/bash ~/anaconda.sh -b -p /opt/conda  \
 && rm ~/anaconda.sh  \
 && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh  \
 && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

RUN conda create -y -n py36 python=3.6 anaconda \
 && echo "conda activate py36" >> ~/.bashrc \
 && /opt/conda/bin/conda install -n py36 -y boost=1.65.1 \
 && /opt/conda/bin/conda install -n py36 -y pytorch-cpu=1.0.1  torchvision-cpu=0.2.2  -c pytorch \
 && /opt/conda/bin/conda install -n py36 -y rdkit=2018.03.1  -c rdkit

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 100 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 100

RUN /opt/conda/envs/py36/bin/pip install torch-scatter==1.1.1 \
 && /opt/conda/envs/py36/bin/pip install  dataclasses arrow

COPY . /electro
RUN unzip  /electro/lef_uspto.zip -d /electro
WORKDIR /electro/notebooks

# Jupyter notebook
EXPOSE 8080

CMD ["./jupyter_notebook_run.sh", "--allow-root", "--ip=0.0.0.0", "--port=8080"]
