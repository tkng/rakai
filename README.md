# Rakai: a strong baseline for multiclass document classification

Rakai is a simple, strong baseline for multiclass classification.
It implements an algorithm called NBSVM, propsed by Wang and Manning at ACL 2012.

## what's difference from original version

## how to build

Rakai is implemented in Go language. Unfortunately, there's no binary distribution, so you have to build from source if you want to use Rakai.

set GOPATH environment to somewhere you want, then following command will generate rakai.

    cd $GOPATH
    mkdir -p src/github.com/tkng
    git clone https://github.com/tkng/rakai.git
    cd rakai/rakai
    go build

I will provide a binary program for Windows and Mac OS X in the future version.

## how to use

Rakai provides three sub commands, say, train, test and predict.

### train

Following procedure will download and train nbsvm. Or you can simply exec ./test.a1a.sh.

    curl http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a > a1a
    ./rakai/rakai train -a nbsvm -m a1a.nbsvm.model -i 10 a1a

  * "-a nbsvm" means you are traning with nbsvm algorithm.
  * -m indicates a filename to store training result
  * "-i 10 " is traning iteration number, say, training loop will executed 10 times
  * last parameter a1a should be libsvm format.

If you want to know more about tuning parameters, see ``rakai train --help''.

### performance test

Following procedure will provide precision, recall and accuracy.

    curl http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t > a1a.t
    ./rakai/rakai test -m a1a.nbsvm.model a1a.t

### predict

Not implemented yet, contribution is welcome!

## experimental results

To be written

## License

Rakai is distributed under MIT license. See LICENSE file for details.

## References

  * Sida Wang and Chris Manning, "Baselines and Bigrams: Simple, Good Sentiment and Text Classification", Proceedings of the ACL, 2012
