package main

import (
	"flag"
	"fmt"
	"github.com/tkng/rakai"
	"log"
	"os"
	"runtime/pprof"
)

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, `
Usage of %s:
   %s [OPTIONS] ARGS...
Options\n`, os.Args[0], os.Args[0])
		flag.PrintDefaults()
	}

	var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
	alg := flag.String("algorithm", "nbsvm", "nbsvm (default) or perceptron")

	flag.Parse()

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	var p rakai.Classifier
	switch *alg {
	case "nbsvm":
		p = rakai.NewNBSVM()
	case "perceptron":
		p = rakai.NewPerceptron()
	default:
		log.Fatal("unsupported algorithm: ", alg)
		return
	}

	if len(flag.Args()) < 2 {
		log.Fatal("unsupported algorithm: ", alg)
	}

	train_filename := flag.Args()[0]
	for i := 0; i < 5; i++ {
		rakai.TrainFile(p, train_filename)
	}

	test_filename := flag.Args()[1]
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
}
