package rakai

import (
	"fmt"
)

type NBSVM struct {
	Labels      *WordManager
	w           []map[string]float64
	count       []map[string]int64
	all_count   map[string]int64
	class_count map[string]int64
	alpha       float64 // smoothness parameter
	eta         float64
}

func NewNBSVM() *NBSVM {
	var nbsvm NBSVM
	nbsvm.w = make([]map[string]float64, 0)
	nbsvm.count = make([]map[string]int64, 0)
	nbsvm.all_count = make(map[string]int64)
	nbsvm.Labels = NewWordManager()
	nbsvm.alpha = 0.1
	nbsvm.eta = 1.0
	return &nbsvm
}

func (nbsvm *NBSVM) update_nb_count(label_id int64, fv []FV) {
	for len(nbsvm.count) < int(label_id)+1 {
		nbsvm.count = append(nbsvm.count, make(map[string]int64))
	}

	for _, x := range fv {
		nbsvm.count[label_id][x.K]++
		nbsvm.all_count[x.K]++
	}
}

func (nbsvm *NBSVM) reweight(label_id int64, fv []FV) []FV {
	new_fv := make([]FV, len(fv))

	for len(nbsvm.count) < int(label_id+1) {
		nbsvm.count = append(nbsvm.count, make(map[string]int64))
	}

	for i, x := range fv {
		c := float64(nbsvm.count[label_id][x.K])
		all := float64(nbsvm.all_count[x.K])
		nb_w := float64((c + nbsvm.alpha) / (all - c + nbsvm.alpha))
		fmt.Println(c, all, nb_w)
		new_fv[i] = FV{x.K, x.V * nb_w}
	}
	fmt.Println("---------")
	fmt.Println(fv)
	fmt.Println(new_fv)
	return new_fv
}

func (p *NBSVM) predict_id(fv []FV) (int64, float64) {
	// TODO: fix -10000.0
	max_score := -10000.0
	id := 0
	for i, w := range p.w {
		scaled_fv := p.reweight(int64(i), fv)
		score := product(w, scaled_fv)
		if score > max_score {
			max_score = score
			id = i
		}
	}
	return int64(id), max_score
}

func (p *NBSVM) predict(fv []FV) (string, float64) {
	id, score := p.predict_id(fv)
	return p.Labels.id2word[id], score
}

func (p *NBSVM) predict_id_all(fv []FV) []float64 {
	ret := make([]float64, 0)

	for _, w := range p.w {
		ret = append(ret, product(w, fv))
	}
	return ret
}

func (p *NBSVM) train1(label string, fv []FV) {
	true_id, ok := p.Labels.word2id[label]

	if !ok {
		p.Labels.add_word(label)
	}
	predicted_id, _ := p.predict_id(fv)

	if predicted_id != true_id {
		p.update_nb_count(true_id, fv)
		p.update_from_id(true_id, fv, 1.0*p.eta)
		p.update_from_id(predicted_id, fv, -1.0*p.eta)
	}
	p.eta *= 0.997
}

func (p *NBSVM) update_from_id(id int64, fv []FV, coeff float64) {
	for len(p.w) < int(id)+1 {
		p.w = append(p.w, make(map[string]float64))
	}

	new_fv := p.reweight(id, fv)

	for i := 0; i < len(new_fv); i++ {
		k := new_fv[i].K
		p.w[id][k] += new_fv[i].V * coeff
	}
}
