FROM gitpod/workspace-full
ENV DEBIAN_FRONTEND=noninteractive

RUN sudo apt-get update -y

RUN sudo apt-get install -y python3.10
