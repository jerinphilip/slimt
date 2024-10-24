#!/bin/bash

sudo apt-get -y install ccache cmake
wget -c --quiet https://dl.google.com/android/repository/android-ndk-r23b-linux.zip
unzip -qq android-ndk-r23b-linux.zip

# Install Java
sudo apt install default-jdk
sudo apt install default-jdk-headless
