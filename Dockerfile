FROM nvidia/cuda:11.3.0-base-ubuntu20.04

WORKDIR /app

RUN apt update
RUN apt-get install -y python3 python3-pip

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install h5py
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install sklearn

RUN apt-get install -y unzip
RUN apt-get install -y wget
RUN mkdir -p data 
RUN wget https://compneuro.net/datasets/shd_test.h5.zip -P data 
RUN wget https://compneuro.net/datasets/shd_train.h5.zip -P data 
RUN cd data && unzip shd_test.h5.zip && rm shd_test.h5.zip
RUN cd data && unzip shd_train.h5.zip && rm shd_train.h5.zip

COPY . .

ENTRYPOINT ["python3", "-u", "learn_curve.py"]



