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

// this is a margin perceptron, not a plain perceptron...

package rakai

import (
	"bufio"
	"fmt"
	"math"
	"os"
)

type Perceptron struct {
	Labels   *WordManager
	Features *WordManager
	w        [][]float64
	eta      float64
	t        int64
}

func NewPerceptron(eta float64) *Perceptron {
	var p Perceptron

	p.Labels = NewWordManager()
	p.Features = NewWordManager()
	p.w = make([][]float64, 0)
	p.eta = eta
	p.t = 0
	return &p
}

func (p *Perceptron) predict_id(fv []FV) (int, float64, int, float64) {
	id := 0
	second_id := 0
	// TODO: fix -10000.0
	max_score := -10000.0
	second_score := -10000.0
	for i, w := range p.w {
		score := product(w, fv)
		if score > max_score {
			second_id = id
			second_score = max_score
			max_score = score
			id = i
		} else if score > second_score {
			second_id = id
			second_score = score
		}
	}
	return id, max_score, second_id, max_score - second_score
}

func (p *Perceptron) predict(fvs []FVS) (string, float64) {
	fv := fvs2fv(p.Features, fvs, false)
	id, score, _, _ := p.predict_id(fv)
	return p.Labels.id2word[id], score
}

func (p *Perceptron) train1(label string, fvs []FVS) {
	true_id, ok := p.Labels.word2id[label]

	if !ok {
		p.Labels.add_word(label)
	}
	fv := fvs2fv(p.Features, fvs, true)
	predicted_id, _, second_id, margin := p.predict_id(fv)

	// lr: learning rate
	lr := math.Pow(p.eta/(1.0+p.eta*float64(p.t)), 0.1)
	if p.t%500 == 0 {
		fmt.Println(predicted_id, true_id, margin, lr)
	}
	if predicted_id != int(true_id) {
		p.update_from_id(true_id, fv, lr)
		p.update_from_id(int64(predicted_id), fv, lr*-1.0)
	} else if margin < 1.0 {
		p.update_from_id(true_id, fv, lr)
		p.update_from_id(int64(second_id), fv, lr*-1.0)
	}
	p.t++
}

func (p *Perceptron) update_from_id(label_id int64, fv []FV, coeff float64) {
	for len(p.w) < int(label_id)+1 {
		p.w = append(p.w, make([]float64, 0))
	}

	for i := 0; i < len(fv); i++ {
		k := fv[i].K
		p.w[label_id] = ensure_w(p.w[label_id], k)
		p.w[label_id][k] += fv[i].V * coeff
	}
}

func (p *Perceptron) Save(filename string) {
	fi, err := os.Create(filename)

	defer func() {
		if err := fi.Close(); err != nil {
			panic(err)
		}
	}()

	if err != nil {
		panic(err)
	}

	writer := bufio.NewWriterSize(fi, 4096*32)

	for label_id, values := range p.w {
		label := p.Labels.id2word[label_id]
		for feature_id, v := range values {
			if v != 0.0 {
				feature := p.Features.id2word[feature_id]
				//				fmt.Fprintf(os.Stderr, "%s\t%s\t%2.4f\n", label, feature, v)
				writer.WriteString(fmt.Sprintf("%s\t%s\t%2.4f\n", label, feature, v))
			}
		}
	}
	writer.Flush()

	// for {
	// 	line, _, err := writer.ReadLine()
	// 	if err == io.EOF {
	// 		break
	// 	}

	// 	label, dat, err := parse_line(string(line))

	// 	if err != nil {
	// 		fmt.Println("err:", err)
	// 	}
	// 	cl.train1(label, dat)
	// }
	//	return nil

}
