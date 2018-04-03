FROM tensorflow/tensorflow:latest-py3

RUN apt-get update && apt-get install -y git
RUN pip install git+https://github.com/n-CLAIR/nsml-local.git
RUN cd / && curl -L -o cli.tar.gz https://github.com/n-CLAIR/File-download/raw/master/nsml/hack/nsml_client.linux.amd64.hack.tar.gz && tar xvpf cli.tar.gz && rm cli.tar.gz

ENV PATH="/client:$PATH"
ENV PYTHONIOENCODING="UTF-8"