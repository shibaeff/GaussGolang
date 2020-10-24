package main

import (
	"fmt"

	"gauss/pkg/gauss"
)

func main() {
	for i := 2; i < 100; i++ {
		s := gauss.NewSystem(gauss.NewRandomMatrix(i, i), gauss.RandFloats(-1000, 1000, i))
		fmt.Println(s.ConditionNumber())
	}
}
