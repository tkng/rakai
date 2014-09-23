package rakai

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// fv: feature and (its) value
type FV struct {
	K string
	V float64
}

type Classifier interface {
	predict_id([]FV) (int64, float64)
	predict([]FV) (string, float64)
	train1(string, []FV)
}

type WordManager struct {
	word2id map[string]int64
	id2word []string
}

func (lm *WordManager) add_word(word string) {
	new_id := int64(len(lm.id2word))
	lm.id2word = append(lm.id2word, word)
	lm.word2id[word] = new_id
}

func NewWordManager() *WordManager {
	var wm WordManager
	wm.word2id = make(map[string]int64)
	wm.id2word = make([]string, 0)
	return &wm
}

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

func product(w map[string]float64, fv []FV) float64 {
	ret := 0.0
	for i := 0; i < len(fv); i++ {
		k := fv[i].K
		ret += w[k] * fv[i].V
	}
	return ret
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

func parse_line(s string) (string, []FV, error) {
	content := make([]FV, 0)
	s = strings.TrimRight(s, "\n")
	s = strings.TrimRight(s, "\r")
	ss := strings.Split(s, " ")
	label := ss[0]
	for i := 1; i < len(ss); i++ {
		e := strings.Split(ss[i], ":")

		if len(e) != 2 {
			if len(e) != 1 {
				// TODO: proper error message generation
				fmt.Println("element size wrong")
				return "", content, errors.New("parse failed")
			}
		} else {
			k := e[0]
			v, _ := strconv.ParseFloat(e[1], 64)
			content = append(content, FV{k, v})
		}
	}
	return label, content, nil
}

func TrainFile(cl Classifier, filename string) error {
	fi, err := os.Open(filename)

	defer func() {
		if err := fi.Close(); err != nil {
			panic(err)
		}
	}()

	if err != nil {
		return err
	}

	reader := bufio.NewReaderSize(fi, 4096*16)
	for {
		line, _, err := reader.ReadLine()
		if err != nil {
			fmt.Println(err)
		}
		if err == io.EOF {
			break
		}

		label, dat, err := parse_line(string(line))

		if err != nil {
			fmt.Println("err:", err)
		}
		cl.train1(label, dat)

		//		fmt.Println(line)
	}
	return nil
}

type stats struct {
	tp int64
	fp int64
	fn int64
}

func TestFile(cl Classifier, filename string) (map[string]stats, error) {
	st := make(map[string]stats)

	fi, err := os.Open(filename)

	defer func() {
		if err := fi.Close(); err != nil {
			panic(err)
		}
	}()

	if err != nil {
		return nil, err
	}

	reader := bufio.NewReaderSize(fi, 4096*16)
	for {
		line, _, err := reader.ReadLine()

		if err == io.EOF {
			break
		}

		label, dat, err := parse_line(string(line))

		predicted, _ := cl.predict(dat)

		s1, ok := st[label]
		if !ok {
			st[label] = stats{}
		}

		if label == predicted {
			s1 := st[label]
			s1.tp += 1
			st[label] = s1
		} else {
			s1.fn += 1
			st[label] = s1

			s2 := st[predicted]
			s2.fp += 1
			st[predicted] = s2
		}
	}

	return st, nil
}

func CalcPrecision(st stats) float64 {
	if st.tp == 0 {
		return 0.0
	}
	return float64(st.tp) / (float64(st.tp) + float64(st.fp))
}

func CalcRecall(st stats) float64 {
	if st.tp == 0 {
		return 0.0
	}
	return float64(st.tp) / (float64(st.tp) + float64(st.fn))
}
