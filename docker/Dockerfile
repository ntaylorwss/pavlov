FROM base_image

COPY jupyter_notebook_config.json /root/.jupyter/

# gym requirements
RUN apt-get update && apt-get install -y zlib1g-dev libjpeg-dev xvfb \
                                         libav-tools xorg-dev python-opengl \
                                         libboost-all-dev libsdl2-dev swig cmake
RUN pip install 'gym[all]'

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

CMD xvfb-run -s '-screen 0 1400x900x24' jupyter notebook --allow-root
RUN echo '#!/bin/bash\necho False > /home/kerasgym/agents/keep_running.txt' \
    > /usr/bin/pause && chmod +x /usr/bin/pause
