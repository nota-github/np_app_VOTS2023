FROM nvcr.io/nvidia/pytorch:22.07-py3

ENV HOME=/root
ENV APP_PATH=$HOME/np_app_VOTS2023
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV PYTHONPATH $APP_PATH
WORKDIR $APP_PATH

# copy
COPY . $APP_PATH/

# pip install
RUN pip install --upgrade pip && \
    sh install_pytorch17.sh
# apt-get install
RUN apt-get update 
RUN apt-get install -y \
    git \ 
    vim \
    curl \ 
    unzip \ 
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 

CMD ["bash"]