#!/bin/sh

if [ ! -f a1a ]; then
  curl http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a > a1a
fi
if [ ! -f a1a.t ]; then
  curl http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t > a1a.t
fi

./rakai/rakai train -a perceptron -m a1a.perceptron.model -i 50 a1a > /dev/null && ./rakai/rakai test -m a1a.perceptron.model a1a.t > a1a.perceptron.result
./rakai/rakai train -a svm -m a1a.svm.model --adagrad=false -i 50 a1a > /dev/null && ./rakai/rakai test -m a1a.svm.model a1a.t > a1a.svm.result
./rakai/rakai train -a nbsvm -m a1a.nbsvm.model -i 50 --adagrad=false a1a > /dev/null && ./rakai/rakai test -m a1a.nbsvm.model a1a.t  > a1a.nbsvm.result

./rakai/rakai train -a svm -m a1a.svm.model -i 50 --adagrad=true a1a > /dev/null && ./rakai/rakai test -m a1a.svm.model a1a.t > a1a.svm.ada.result
./rakai/rakai train -a nbsvm -m a1a.nbsvm.model -i 50 --adagrad=true a1a > /dev/null && ./rakai/rakai test -m a1a.nbsvm.model a1a.t > a1a.nbsvm.ada.result
