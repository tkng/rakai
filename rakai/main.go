package main

import (
	"flag"
	"fmt"
	"github.com/tkng/rakai"
	"log"
	"os"
	"runtime/pprof"
	//	"strings"
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

	for _, filename := range flag.Args() {
		for i := 0; i < 10; i++ {
			fmt.Println("----------")
			rakai.TrainFile(p, filename)
			st, err := rakai.TestFile(p, filename)
			for label, x := range st {
				fmt.Println(label)
				fmt.Println("  ", rakai.CalcPrecision(x))
				fmt.Println("  ", rakai.CalcRecall(x))
			}
			if err != nil {
				fmt.Println(err)
			}
		}
	}
}
