FROM floydhub/pytorch:0.3.0-gpu.cuda8cudnn6-py3.17

RUN apt-get update && apt-get install -y unzip
RUN curl -L -o master.zip https://github.com/facebookresearch/fastText/archive/master.zip && \
      unzip master.zip && rm master.zip && mv fastText-master fastText && \
      cd fastText && make

RUN apt-get update && apt-get install -y python3-dev cmake gcc
RUN cd fastText && pip install .

ENV PATH /fastText:$PATH
ENV PYTHONPATH=/fastText:$PYTHONPATH
