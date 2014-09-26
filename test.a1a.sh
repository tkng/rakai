#!/bin/sh

if [ ! -f a1a ]; then
  curl http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a > a1a
fi
if [ ! -f a1a.t ]; then
  curl http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t > a1a.t
fi

./rakai/rakai train -a perceptron -m a1a.perceptron.model -i 10 a1a > /dev/null && ./rakai/rakai test -m a1a.perceptron.model a1a.t
./rakai/rakai train -a nbsvm -m a1a.nbsvm.model -i 10 a1a > /dev/null && ./rakai/rakai test -m a1a.nbsvm.model a1a.t
