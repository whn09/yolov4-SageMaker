ARG REGISTRY_URI

# FROM ${REGISTRY_URI}/pytorch-inference:1.5.0-cpu-py36-ubuntu16.04

FROM ${REGISTRY_URI}/pytorch-inference:1.5.0-gpu-py36-cu101-ubuntu16.04
RUN rm -rf /etc/apt/sources.list.d/*

RUN mkdir -p /opt/ml/model

# COPY package/ /opt/ml/code/package/

# COPY serve.py /opt/ml/model/code/

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

##########################################################################################
# SageMaker requirements
##########################################################################################
## install flask
RUN pip install networkx==2.3 flask gevent gunicorn boto3 opencv-python==4.4.0.40 -i https://opentuna.cn/pypi/web/simple

WORKDIR /opt/program
RUN apt-get -y update && apt-get install -y build-essential cmake unzip pkg-config libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran python3-dev
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.4.0.zip
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.4.0.zip
RUN unzip opencv.zip
RUN unzip opencv_contrib.zip
RUN mv opencv-4.4.0 opencv
RUN mv opencv_contrib-4.4.0 opencv_contrib
RUN pip install numpy -i https://opentuna.cn/pypi/web/simple
WORKDIR /opt/program/opencv
RUN mkdir build
WORKDIR /opt/program/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D INSTALL_C_EXAMPLES=OFF \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D WITH_CUDA=ON \
	-D WITH_CUDNN=ON \
	-D OPENCV_DNN_CUDA=ON \
	-D ENABLE_FAST_MATH=1 \
	-D CUDA_FAST_MATH=1 \
	-D CUDA_ARCH_BIN=10.1 \
	-D WITH_CUBLAS=1 \
	-D OPENCV_EXTRA_MODULES_PATH=/opt/program/opencv_contrib/modules \
	-D HAVE_opencv_python3=ON \
	-D PYTHON_EXECUTABLE=python \
	-D BUILD_EXAMPLES=ON ..
RUN make -j8 & make install & ldconfig

### Install nginx notebook
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# forward request and error logs to docker log collector
RUN ln -sf /dev/stdout /var/log/nginx/access.log
RUN ln -sf /dev/stderr /var/log/nginx/error.log

# Set up the program in the image
COPY * /opt/program/
COPY ./pretrained_model/* /opt/program/
WORKDIR /opt/program

ENTRYPOINT ["python", "serve.py"]

