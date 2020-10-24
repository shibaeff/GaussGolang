package generate

import "gauss/pkg/gauss"

type Generator interface {
	Gen(n int) *gauss.System
}

type gen struct {
	matrixLambda func(i int, j int) float64
	rowLambda    func(i int) float64
}

func (g *gen) Gen(n int) *gauss.System {
	matrix := make([][]float64, n)
	row := make([]float64, n)
	for i := 0; i < n; i++ {
		row[i] = g.rowLambda(i)
		matrix[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			matrix[i][j] = g.matrixLambda(i, j)
		}
	}
	return &gauss.System{
		B: row,
		M: gauss.NewMatrixFromArr(matrix),
	}
}
