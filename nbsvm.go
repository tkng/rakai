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
	nb_w := (c + alpha) / (c2 + alpha) / ((all - c + alpha) / (all2 - c2 + alpha))
	return math.Sqrt(nb_w)
}

func (nbsvm *NBSVM) reweight(label_id int64, fv []FV) []FV {
	new_fv := make([]FV, len(fv))

	for len(nbsvm.count) < int(label_id)+1 {
		nbsvm.count = append(nbsvm.count, make([]int64, 0))
	}

	for i, x := range fv {
		nb_w := calc_weight(nbsvm, label_id, x.K)
		//		fmt.Println(nb_w, math.Log(nb_w+1), math.Log(nb_w), math.Sqrt(nb_w), math.Sqrt(nb_w+1))
		//		nb_w = math.Sqrt(nb_w)

		// fmt.Println("----")
		// fmt.Println(c, c2, all, all2)
		// fmt.Println(c+alpha, c2+alpha, all-c+alpha, all2-c2+alpha)
		// fmt.Println((c+alpha)/(c2+alpha), (all+alpha)/(all2+alpha),
		// 	(all-c+alpha)/(all2-c2+alpha))

		if nb_w < 1 {
			//			fmt.Println(nb_w, nb_w*0.25+0.75)
			nb_w = nb_w*0.25 + 0.75
		} else {
			//			fmt.Println(nb_w, math.Sqrt(nb_w))
			nb_w = math.Sqrt(nb_w)
		}

		//		nb_w = math.Log(nb_w+1)

		new_fv[i] = FV{x.K, x.V * nb_w}
	}
	//	fmt.Println("---------")
	//	fmt.Println(fv)
	//	fmt.Println(new_fv)
	return new_fv
}

func (p *NBSVM) predict_id(fv []FV) (int, float64, int, float64) {
	//	p.regularize_l1(fv)

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

func clip(v float64, lu float64, lambda float64, t int64) float64 {
	//	fmt.Println("--")
	//	fmt.Println(v, lu, lambda, t, (float64(t)-lu)*lambda)
	if v > 0.0 {
		if v > (float64(t)-lu)*lambda {
			fmt.Println(v, v-(float64(t)-lu)*lambda, (float64(t) - lu))
			return v - (float64(t)-lu)*lambda
		} else {
			//			fmt.Println(0.0)
			return 0.0
		}
	} else if v < 0.0 {
		if v < (float64(t)-lu)*lambda {
			//			fmt.Println(vgu, v+(float64(t)-lu)*lambda)
			return v + (float64(t)-lu)*lambda
		} else {
			//			fmt.Println(0.0)
			return 0.0
		}
	}
	return 0.0
}

func (p *NBSVM) regularize_l1(fv []FV) {
	for class, _ := range p.w {
		for _, x := range fv {
			//			fmt.Println(len(p.w[class]), len(p.lu[class]), x.K)
			if int(x.K) < len(p.w[class]) {
				//				fmt.Println(p.w[class][x.K], x.K)
				p.w[class][x.K] = clip(p.w[class][x.K], p.lu[class][x.K], p.lambda, p.t)
				//				fmt.Println(p.w[class][x.K], x.K)

			} else {
				for len(p.lu[class]) < int(x.K)+1 {
					p.lu[class] = append(p.lu[class], float64(p.t))
				}
			}
			p.lu[class][x.K] = float64(p.t)
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
		p.update_from_id(true_id, fv, lr)
		p.update_from_id(int64(predicted_id), fv, lr*-1.0)
	} else if margin < 100.0 {
		p.update_from_id(true_id, fv, lr)
		p.update_from_id(int64(second_id), fv, lr*-1.0)
	}
	p.t++
}

func ensure_lu(lu []float64, k int64) []float64 {
	for len(lu) < int(k)+1 {
		lu = append(lu, 0.0)
	}
	return lu
}

func (p *NBSVM) update_from_id(label_id int64, fv []FV, coeff float64) {
	for len(p.w) < int(label_id)+1 {
		p.w = append(p.w, make([]float64, 0))
		p.lu = append(p.lu, make([]float64, 0.0))
	}
	new_fv := p.reweight(label_id, fv)

	for i := 0; i < len(new_fv); i++ {
		k := new_fv[i].K

		p.w[label_id] = ensure_w(p.w[label_id], k)
		p.lu[label_id] = ensure_lu(p.lu[label_id], k)
		p.w[label_id][k] += new_fv[i].V * coeff
	}
}

func (p *NBSVM) Save(filename string) {

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
