ARG baseimage
FROM $baseimage

# gym requirements
RUN apt-get update && apt-get install -y zlib1g-dev libjpeg-dev xvfb \
                                         libav-tools xorg-dev python-opengl python3-tk \
                                         libboost-all-dev libsdl2-dev swig cmake

RUN pip install 'gym[all]'

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN rm -f /usr/bin/python && ln -s /usr/bin/python3.6 /usr/bin/python

RUN useradd -ms /bin/bash pavlov
USER pavlov

RUN mkdir /home/pavlov/.jupyter && \
    echo "c.NotebookApp.token = u''" > /home/pavlov/.jupyter/jupyter_notebook_config.py
USER root
RUN chown pavlov:pavlov /etc/jupyter/jupyter_notebook_config.py
USER pavlov

CMD xvfb-run -s '-screen 0 1400x900x24' jupyter notebook
WORKDIR /home/pavlov

# exclusive to public version
COPY pavlov/ /usr/local/lib/python3.6/dist-packages/pavlov
COPY tests/ /tmp
RUN python -m pytest /tmp
