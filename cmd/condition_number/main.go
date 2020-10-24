package main

import (
	"fmt"

	"gauss/pkg/gauss"
)

func main() {
	m := gauss.NewMatrixFromArr([][]float64{
		{5, 7},
		{7, 10},
	})
	b := []float64{11, 1101}
	s := gauss.NewSystem(m, b)
	fmt.Println(s.ConditionNumber())
}
