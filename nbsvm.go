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

package rakai

import (
	"bufio"
	"fmt"
	"math"
	"os"
)

type NBSVM struct {
	Labels          *WordManager
	Features        *WordManager
	w               [][]float64
	lu              [][]float64 // last update
	count           [][]int64
	all_count       []int64
	class_count     []int64
	class_count_all int64
	alpha           float64 // smoothness parameter
	eta             float64
	t               int64
	lambda          float64
	ada             [][]float64
}

func NewNBSVM(alpha float64, eta float64, lambda float64) *NBSVM {
	var nbsvm NBSVM
	nbsvm.Labels = NewWordManager()
	nbsvm.Features = NewWordManager()
	nbsvm.ada = make([][]float64, 0)
	nbsvm.w = make([][]float64, 0)
	nbsvm.lu = make([][]float64, 0)
	nbsvm.ada = make([][]float64, 0)
	nbsvm.count = make([][]int64, 0)
	nbsvm.all_count = make([]int64, 0)
	nbsvm.class_count = make([]int64, 0)
	nbsvm.class_count_all = 0
	nbsvm.alpha = alpha
	nbsvm.eta = eta
	nbsvm.t = 0
	nbsvm.lambda = lambda
	return &nbsvm
}

func (nbsvm *NBSVM) update_nb_count(label_id int64, fv []FV) {
	for len(nbsvm.count) < int(label_id)+1 {
		nbsvm.count = append(nbsvm.count, make([]int64, 0))
	}

	for len(nbsvm.class_count) < int(label_id)+1 {
		nbsvm.class_count = append(nbsvm.class_count, 0)
	}

	nbsvm.class_count[label_id]++
	nbsvm.class_count_all++

	for _, x := range fv {
		for len(nbsvm.count[label_id]) < int(x.K)+1 {
			nbsvm.count[label_id] = append(nbsvm.count[label_id], 0.0)
		}
		for len(nbsvm.all_count) < int(x.K)+1 {
			nbsvm.all_count = append(nbsvm.all_count, 0.0)
		}
		nbsvm.count[label_id][x.K]++
		nbsvm.all_count[x.K]++
	}
}

func calc_weight(nbsvm *NBSVM, label_id, feature_id int64) float64 {
	alpha := nbsvm.alpha

	c := 0.0
	all := 0.0
	c2 := 0.0
	all2 := float64(nbsvm.class_count_all)

	if int(feature_id) < len(nbsvm.count[label_id]) {
		c = float64(nbsvm.count[label_id][feature_id])
	}
	if int(feature_id) < len(nbsvm.all_count) {
		all = float64(nbsvm.all_count[feature_id])
	}
	if int(label_id) < len(nbsvm.class_count) {
		c2 = float64(nbsvm.class_count[label_id])
	}
	nb_w := (c + alpha) / (c2 + alpha + c2*alpha) / ((all - c + alpha) / (all2 - c2 + alpha + (all2-c2)*alpha))
	//	nb_w := (c + alpha) / (c2 + alpha) / ((all - c + alpha) / (all2 - c2 + alpha))

	// if nb_w > 1.0 {
	// 	return math.Sqrt(nb_w)
	// } else {
	// 	return math.Pow(nb_w, 0.1)
	// }

	//	return math.Sqrt(nb_w)

	return math.Log(nb_w + 1)
}

func (nbsvm *NBSVM) reweight(label_id int64, fv []FV) []FV {
	new_fv := make([]FV, len(fv))

	for len(nbsvm.count) < int(label_id)+1 {
		nbsvm.count = append(nbsvm.count, make([]int64, 0))
	}

	for i, x := range fv {
		nb_w := calc_weight(nbsvm, label_id, x.K)
		new_fv[i] = FV{x.K, x.V * nb_w}
	}
	return new_fv
}

func (p *NBSVM) predict_id(fv []FV) (int, float64, int, float64) {
	p.regularize_l1(fv)

	id := 0
	second_id := 0
	// TODO: fix -10000.0
	max_score := -100000.0
	second_score := -100000.0
	for i, w := range p.w {
		scaled_fv := p.reweight(int64(i), fv)
		score := product(w, scaled_fv)
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

func (p *NBSVM) predict(fvs []FVS) (string, float64) {
	fv := fvs2fv(p.Features, fvs, false)
	id, score, _, _ := p.predict_id(fv)
	return p.Labels.id2word[id], score
}

func (p *NBSVM) clip(v float64, lu float64, lambda float64, t int64) float64 {
	if v > 0.0 {
		if v > (float64(t)-lu)*lambda {
			return v - (float64(t)-lu)*lambda
		} else {
			return 0.0
		}
	} else if v < 0.0 {
		if v < (float64(t)-lu)*lambda {
			return v + (float64(t)-lu)*lambda
		} else {
			return 0.0
		}
	}
	return 0.0
}

func (p *NBSVM) regularize_l1(fv []FV) {
	for label_id, _ := range p.w {
		for _, x := range fv {
			feature_id := x.K
			if int(feature_id) < len(p.w[label_id]) {
				lu := p.lu[label_id][feature_id]
				lr := p.calc_learning_rate(int64(label_id), int64(feature_id))
				p.w[label_id][feature_id] = p.clip(p.w[label_id][feature_id], lu, p.lambda*lr, p.t)
			} else {
				for len(p.lu[label_id]) < int(x.K)+1 {
					p.lu[label_id] = append(p.lu[label_id], float64(p.t))
				}
			}
			p.lu[label_id][x.K] = float64(p.t)
		}
	}
}

func (p *NBSVM) regularize_l1_all() {
	for label_id, _ := range p.w {
		for feature_id, v := range p.w[label_id] {
			lu := p.lu[label_id][feature_id]
			lr := p.calc_learning_rate(int64(label_id), int64(feature_id))
			p.w[label_id][feature_id] = p.clip(v, lu, p.lambda*lr, p.t)
			p.lu[label_id][feature_id] = float64(p.t)
		}
	}
}

func (p *NBSVM) train1(label string, fvs []FVS) {
	true_id, ok := p.Labels.word2id[label]

	if !ok {
		p.Labels.add_word(label)
	}
	fv := fvs2fv(p.Features, fvs, true)
	predicted_id, _, second_id, margin := p.predict_id(fv)

	lr := math.Pow(p.eta/(1.0+p.eta*float64(p.t)), 0.01)
	//	lr := p.eta / (1.0 + math.Sqrt(p.eta*float64(p.t)))
	if p.t%500 == 0 {
		fmt.Printf("%v\t%v\t%4.2f\t%2.6f\n", predicted_id, true_id, margin, lr)
	}

	p.update_nb_count(true_id, fv)
	if predicted_id != int(true_id) {
		p.update_from_id(true_id, fv, 1.0)
		p.update_from_id(int64(predicted_id), fv, -1.0)
	} else if margin < 1.0 {
		p.update_from_id(true_id, fv, 1.0)
		p.update_from_id(int64(second_id), fv, -1.0)
	}
	p.t++
}

func ensure_lu(lu []float64, k int64) []float64 {
	for len(lu) < int(k)+1 {
		lu = append(lu, 0.0)
	}
	return lu
}

func (p *NBSVM) calc_learning_rate(label_id, feature_id int64) float64 {
	return 1.0 / math.Sqrt(p.ada[label_id][feature_id]+1.0)
	//	return 1.0 / math.Pow(p.ada[label_id][feature_id]+1.0, 0.25)
}

func (p *NBSVM) update_from_id(label_id int64, fv []FV, coeff float64) {
	for len(p.w) < int(label_id)+1 {
		p.w = append(p.w, make([]float64, 0))
		p.lu = append(p.lu, make([]float64, 0.0))
		p.ada = append(p.ada, make([]float64, 0.0))
	}
	new_fv := p.reweight(label_id, fv)

	for i := 0; i < len(new_fv); i++ {
		k := new_fv[i].K

		p.w[label_id] = ensure_w(p.w[label_id], k)
		p.lu[label_id] = ensure_lu(p.lu[label_id], k)
		p.ada[label_id] = ensure_lu(p.ada[label_id], k)
		lr := p.calc_learning_rate(label_id, k)
		delta := new_fv[i].V * coeff * lr

		p.w[label_id][k] += delta
		p.ada[label_id][k] += delta * delta
	}
}

func (p *NBSVM) Save(filename string) {
	p.regularize_l1_all()
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
				v *= calc_weight(p, int64(label_id), int64(feature_id))
				writer.WriteString(fmt.Sprintf("%s\t%s\t%2.4f\n", label, feature, v))
			}
		}
	}
	writer.Flush()
}
