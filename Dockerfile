# humble desktop full
from osrf/ros:humble-desktop-full

RUN apt update
RUN apt install python3-vcstool python3-pip python3-rosdep python3-colcon-common-extensions -y


WORKDIR /topological_localization