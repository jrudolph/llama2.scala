#!/bin/bash

gcc -fopenmp -fPIC -O3 -march=native \
  -I/usr/lib/jvm/java-17-openjdk-amd64/include \
  -I/usr/lib/jvm/java-17-openjdk-amd64/include/linux \
  -shared -o libmatmul.so \
  vectmult.c