#!/bin/bash

wget -c --quiet https://dl.google.com/android/repository/android-ndk-r23b-linux.zip
unzip -qq android-ndk-r23b-linux.zip
sudo apt-get -y install ccache cmake
