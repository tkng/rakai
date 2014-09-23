package rakai

type Perceptron struct {
	Labels   *WordManager
	Features *WordManager
	w        []map[string]float64
	eta      float64
}

func NewPerceptron() *Perceptron {
	var p Perceptron

	p.Labels = NewWordManager()
	p.Features = NewWordManager()
	p.w = make([]map[string]float64, 0)
	p.eta = 1.0
	return &p
}

func (p *Perceptron) predict_id(fv []FV) (int64, float64) {
	// TODO: fix -10000.0
	max_score := -10000.0
	id := 0
	for i, w := range p.w {
		score := product(w, fv)
		if score > max_score {
			max_score = score
			id = i
		}
	}
	return int64(id), max_score
}

func (p *Perceptron) predict(fv []FV) (string, float64) {
	id, score := p.predict_id(fv)
	return p.Labels.id2word[id], score
}

func (p *Perceptron) predict_id_all(fv []FV) []float64 {
	ret := make([]float64, 0)

	for _, w := range p.w {
		ret = append(ret, product(w, fv))
	}
	return ret
}

func (p *Perceptron) train1(label string, fv []FV) {
	true_id, ok := p.Labels.word2id[label]

	if !ok {
		p.Labels.add_word(label)
	}
	id, _ := p.predict_id(fv)

	if id != true_id {
		p.update_from_id(true_id, fv, 1.0*p.eta)
		p.update_from_id(id, fv, -1.0*p.eta)
	}
	p.eta *= 0.997
}

func (p *Perceptron) update_from_id(id int64, fv []FV, coeff float64) {
	for len(p.w) < int(id)+1 {
		p.w = append(p.w, make(map[string]float64))
	}

	for i := 0; i < len(fv); i++ {
		k := fv[i].K
		p.w[id][k] += fv[i].V * coeff
	}
}
