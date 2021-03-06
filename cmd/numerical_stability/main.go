package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gauss/pkg/gauss"
)

const (
	N         = 1000
	lower     = 2
	trialsNum = 1
)

func perturb(arr []float64, err float64) (ret []float64) {
	ret = make([]float64, len(arr))
	copy(ret, arr)
	//for i := 0; i < len(arr); i++ {
	//	ret[i] += err
	//}
	ret[0] += err
	return
}

func dist(x []float64, y []float64) (ans float64) {
	for i := 0; i < len(x); i++ {
		ans += (x[i] - y[i]) * (x[i] - y[i])
	}
	ans = math.Sqrt(ans)
	return
}

func norm(x []float64) (ans float64) {
	for i := 0; i < len(x); i++ {
		ans += x[i] * x[i]
	}
	ans = math.Sqrt(ans)
	return
}

func main() {
	err := 0.01
	var err_vec []float64
	for trial := 0; trial < trialsNum; trial++ {
		for n := lower; n < N; n *= 2 {
			rand.Seed(int64(time.Now().Second()))
			rhs := gauss.RandFloats(-1000, 1000, n)
			matrix := gauss.NewRandomMatrix(n, n)
			old := matrix.Copy()
			system := gauss.NewSystem(matrix, rhs)
			x, _ := system.GaussSolve()
			// fmt.Println(x)
			matrix_copy := old.Copy()
			// fmt.Println(matrix_copy.Arr)
			new_rhs := perturb(rhs, err)
			new_system := gauss.NewSystem(matrix_copy, new_rhs)
			x_star, _ := new_system.GaussSolve()
			d := dist(x, x_star)
			norm := norm(x)
			err_vec = append(err_vec, d/norm)
		}
	}
	for i := 0; i < len(err_vec); i++ {
		err_vec[i] /= trialsNum
		// fmt.Println(err_vec[i])
	}
	fmt.Println(err_vec)
	// fmt.Println(perturb([]float64{1, 2, 3}, 0.1))
	//x_values := make([]float64, N)
	//for i := 0; i < N; i++ {
	//	x_values[i] = float64(i)
	//}
	//graph := chart.Chart{
	//	Series: []chart.Series{
	//		chart.ContinuousSeries{
	//			XValues: x_values,
	//			YValues: err_vec,
	//		},
	//	},
	//}
	//
	//buffer := bytes.NewBuffer([]byte{})
	//graph.Render(chart.PNG, buffer)
	//ioutil.WriteFile("chart.png", buffer.Bytes(), 0600)
}
