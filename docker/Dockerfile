ARG baseimage
FROM $baseimage

# gym requirements
RUN apt-get update && apt-get install -y zlib1g-dev libjpeg-dev xvfb \
                                         libav-tools xorg-dev python-opengl \
                                         libboost-all-dev libsdl2-dev swig cmake

RUN pip install 'gym[all]'

ARG reqfile
COPY $reqfile /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

CMD xvfb-run -s '-screen 0 1400x900x24' jupyter notebook --allow-root
RUN echo '#!/bin/bash\necho False > /home/pavlov/agents/keep_running.txt' \
    > /usr/bin/pause && chmod +x /usr/bin/pause

RUN mkdir /root/.jupyter && \
    echo "c.NotebookApp.token = u''" > /root/.jupyter/jupyter_notebook_config.py
