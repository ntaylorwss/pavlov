FROM tensorflow_gpu
COPY jupyter_notebook_config.json /root/.jupyter/jupyter_notebook_config.json

# gym requirements
RUN apt-get update && apt-get install -y zlib1g-dev libjpeg-dev xvfb \
                                         libav-tools xorg-dev python-opengl \
                                         libboost-all-dev libsdl2-dev swig cmake
RUN pip install 'gym[all]'

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

