// Copyright (c) 2014 TOKUNAGA Hiroyuki

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package main

import (
	"flag"
	"fmt"
	"github.com/tkng/rakai"
	"log"
	"os"
)

func train_file(args []string) {
	var (
		adagrad        bool
		algorithm      string
		model_filename string
		iterations     int
	)
	fmt.Println(args)
	fs := flag.NewFlagSet("train", flag.ExitOnError)
	fs.StringVar(&algorithm, "algorithm", "nbsvm", "algorithm for training , nbsvm (default) or perceptron")
	fs.StringVar(&algorithm, "a", "nbsvm", "algorithm for training , nbsvm (default) or perceptron")
	fs.BoolVar(&adagrad, "adagrad", true, "enable adagrad")
	fs.StringVar(&model_filename, "model", "", "model filename")
	fs.StringVar(&model_filename, "m", "", "model filename")

	alpha := fs.Float64("alpha", 0.01, "additive parameter")
	eta := fs.Float64("eta", 0.1, "initial learning rate")
	lambda := fs.Float64("lambda", 1.0e-8, "regularization parameter")
	fs.IntVar(&iterations, "iterations", 10, "iteration number")
	fs.IntVar(&iterations, "i", 10, "iteration number")

	fs.Parse(args)

	// FIXME: model filename check

	var p rakai.Classifier
	switch algorithm {
	case "nbsvm":
		p = rakai.NewNBSVM(*alpha, *eta, *lambda, adagrad)
	case "svm":
		p = rakai.NewSVM(*eta, *lambda, adagrad)
	case "perceptron":
		p = rakai.NewPerceptron(*eta)
	default:
		log.Fatal("unsupported algorithm: ", algorithm)
		return
	}

	for _, train_filename := range fs.Args() {
		fmt.Println(train_filename)

		for i := 0; i < iterations; i++ {
			rakai.TrainFile(p, train_filename)
		}
	}
	p.Save(model_filename)
}

func test_file(args []string) {
	var (
		model_filename string
	)

	fs := flag.NewFlagSet("test", flag.ExitOnError)
	fs.StringVar(&model_filename, "model", "", "model filename")
	fs.StringVar(&model_filename, "m", "", "model filename")

	fs.Parse(args)

	p := rakai.NewPredictor(model_filename)

	test_filename := fs.Args()[0]
	st, err := rakai.TestFile(p, test_filename)
	if err != nil {
		fmt.Println(err)
	}
	for _, label := range rakai.Mapkeys(st) {
		fmt.Println(label)
		fmt.Println("  ", rakai.CalcPrecision(st[label]))
		fmt.Println("  ", rakai.CalcRecall(st[label]))
		fmt.Println("  ", st[label])
	}
	acc, nt, nf := rakai.CalcAccuracy(st)
	fmt.Println("acc:", acc, nt, nf)
}

func predict(args []string) {
	fmt.Println("sorry, predict is not implemented yet")
	os.Exit(1)
}

var usage = `
Usage %s <Command> [Options]

Commands:
  train   train model
  test    test and caluculate precision, recall, accuracy
  predict predict
`

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, usage, os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()
	args := flag.Args()

	if len(args) == 0 {
		flag.Usage()
		os.Exit(1)
	}

	switch args[0] {
	case "train":
		train_file(args[1:])
	case "test":
		test_file(args[1:])
	case "predict":
		predict(args[1:])
	default:
		flag.Usage()
		os.Exit(1)
	}
}
