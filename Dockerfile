FROM tensorflow/tensorflow:1.4.1-py3

RUN apt-get update && apt-get install -y git
RUN pip install git+https://github.com/n-CLAIR/nsml-local.git
RUN cd / && curl -L -o cli.tar.gz https://github.com/n-CLAIR/File-download/raw/master/nsml/hack/nsml_client.linux.amd64.hack.tar.gz && tar xvpf cli.tar.gz && rm cli.tar.gz

ENV PATH="/client:$PATH"
ENV PYTHONIOENCODING="UTF-8"

RUN pip install keras
RUN pip install http://download.pytorch.org/whl/cpu/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl
RUN pip install torchvision
RUN pip install numpy==1.13.3
RUN pip install pandas==0.20.3