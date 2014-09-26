package rakai

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
)

// fv: feature and (its) value
type FV struct {
	K int64
	V float64
}

// fvs: feature and (its) value
type FVS struct {
	K string
	V float64
}

type Classifier interface {
	// returns label_id, score, margin
	predict_id([]FV) (int, float64, int, float64)
	predict([]FVS) (string, float64)
	train1(string, []FVS)
	Save(string)
}

type Predictor struct {
	Labels   *WordManager
	Features *WordManager
	w        [][]float64
}

type WordManager struct {
	word2id map[string]int64
	id2word []string
}

func (wm *WordManager) add_word(word string) int64 {
	new_id := int64(len(wm.id2word))
	wm.id2word = append(wm.id2word, word)
	wm.word2id[word] = new_id
	return new_id
}

func (wm *WordManager) get_word(word string, update bool) int64 {
	id, ok := wm.word2id[word]

	if ok {
		return id
	} else if update {
		return wm.add_word(word)
	}
	return -1
}

func NewWordManager() *WordManager {
	var wm WordManager
	wm.word2id = make(map[string]int64)
	wm.id2word = make([]string, 0)
	return &wm
}

func fvs2fv(wm *WordManager, fvs []FVS, update bool) []FV {
	ret := make([]FV, 0, len(fvs))
	for _, x := range fvs {
		k, ok := wm.word2id[x.K]
		if ok {
			ret = append(ret, FV{k, x.V})
		} else if update {
			k = wm.add_word(x.K)
			ret = append(ret, FV{k, x.V})
		}
	}
	return ret
}

func ensure_w(w []float64, k int64) []float64 {
	for len(w) < int(k)+1 {
		w = append(w, 0.0)
	}
	return w
}

func product(w []float64, fv []FV) float64 {
	ret := 0.0
	for i := 0; i < len(fv); i++ {
		k := fv[i].K
		if len(w) < int(k)+1 {
			continue
		}
		ret += w[k] * fv[i].V
	}
	return ret
}

func parse_line(s string) (string, []FVS, error) {
	content := make([]FVS, 0)
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
			content = append(content, FVS{k, v})
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

	reader := bufio.NewReaderSize(fi, 4096*64)
	for {
		line, _, err := reader.ReadLine()
		if err == io.EOF {
			break
		}

		label, dat, err := parse_line(string(line))

		if err != nil {
			fmt.Println("err:", err)
		}
		cl.train1(label, dat)
	}
	return nil
}

type stats struct {
	tp int64
	fp int64
	fn int64
}

func TestFile(cl *Predictor, filename string) (map[string]stats, error) {
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

	reader := bufio.NewReaderSize(fi, 4096*64)
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

func (p *Predictor) predict_id(fv []FV) (int, float64, int, float64) {
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

func (p *Predictor) predict(fvs []FVS) (string, float64) {
	fv := fvs2fv(p.Features, fvs, false)
	id, score, _, _ := p.predict_id(fv)
	return p.Labels.id2word[id], score
}

func NewPredictor(filename string) *Predictor {
	var p Predictor

	p.Labels = NewWordManager()
	p.Features = NewWordManager()
	p.w = make([][]float64, 0)

	fi, err := os.Open(filename)

	if err != nil {
		fmt.Println(err)
	}

	reader := bufio.NewReaderSize(fi, 4096*64)
	for {
		line, _, err := reader.ReadLine()

		if err == io.EOF {
			break
		}

		s := strings.TrimRight(string(line), "\n")
		s = strings.TrimRight(s, "\r")
		ss := strings.Split(s, "\t")

		if len(ss) != 3 {
			fmt.Println("model file format error")
			os.Exit(1)
		}

		label := ss[0]
		feature := ss[1]
		v, _ := strconv.ParseFloat(ss[2], 64)
		label_id := p.Labels.get_word(label, true)
		feature_id := p.Features.get_word(feature, true)
		//		fmt.Println(label_id, feature_id, v)
		add_weight(&p, label_id, feature_id, v)
	}

	return &p
}

func add_weight(p *Predictor, label_id int64, feature_id int64, v float64) {
	for len(p.w) < int(label_id)+1 {
		p.w = append(p.w, make([]float64, 0))
	}
	for len(p.w[label_id]) < int(feature_id)+1 {
		p.w[label_id] = append(p.w[label_id], 0.0)
	}
	p.w[label_id][feature_id] = v
}

func Mapkeys(m map[string]stats) []string {
	vec := make([]string, 0)
	for k, _ := range m {
		vec = append(vec, k)
	}
	sort.Strings(vec)
	return vec
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

func CalcAccuracy(m map[string]stats) (float64, int, int) {
	num_true := 0
	num_false := 0

	for _, v := range m {
		num_true += int(v.tp)
		num_false += int(v.fp)
		num_false += int(v.fn)
	}
	acc := float64(num_true) / (float64(num_false)/2.0 + float64(num_true))
	return acc, num_true, num_false
}
