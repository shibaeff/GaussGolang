package gauss

import (
	"math"
	"math/rand"
)

const (
	minFloat = -1000
	maxFloat = 1000

	eps = 0.001
)

type Matrix struct {
	Arr [][]float64
}

func (m *Matrix) Norm() (norm float64) {
	for _, r := range m.Arr {
		for _, item := range r {
			norm += item * item
		}
	}
	norm = math.Sqrt(norm)
	return
}

func (m *Matrix) Copy() (copy *Matrix) {
	matrix := make([][]float64, len(m.Arr))
	for i := 0; i < len(m.Arr[0]); i++ {
		matrix[i] = append([]float64(nil), m.Arr[i]...)
	}
	return &Matrix{
		Arr: matrix,
	}
}

func RandFloats(min, max float64, n int) []float64 {
	res := make([]float64, n)
	for i := range res {
		res[i] = min + rand.Float64()*(max-min)
	}
	return res
}

func (m *Matrix) Equal(another *Matrix) bool {
	for i, _ := range m.Arr {
		for j, _ := range m.Arr[i] {
			if math.Abs(m.Arr[i][j]-another.Arr[i][j]) > eps {
				return false
			}
		}
	}
	return true
}

func (m *Matrix) IsLowerTriangular() bool {
	N := len(m.Arr)
	// M := len(m.Arr[0])
	for i := 1; i < N; i++ {
		for j := 0; j < i; j++ {
			if m.Arr[i][j] != 0 {
				return false
			}
		}
	}
	return true
}

// Extend позволяет расширить матрицу другой матрицей
func (m *Matrix) Extend(another *Matrix) {
	for i, _ := range m.Arr {
		m.Arr[i] = append(m.Arr[i], another.Arr[i]...)
	}
}

// Submatrix позволяет взять подматрицу
func (m *Matrix) Submatrix() (new *Matrix) {
	n := len(m.Arr)
	matrix := make([][]float64, n)
	for i := 0; i < n; i++ {
		matrix[i] = make([]float64, n)
	}

	for i := 0; i < n; i++ {
		copy(matrix[i], m.Arr[i][n:])
	}
	return &Matrix{Arr: matrix}
}

func NewMatrixFromArr(arr [][]float64) (m *Matrix) {
	return &Matrix{
		Arr: arr,
	}
}

func NewRandomMatrix(m, n int) *Matrix {
	matrix := make([][]float64, m)
	for i := 0; i < m; i++ {
		matrix[i] = RandFloats(minFloat, maxFloat, n)
	}
	return &Matrix{
		Arr: matrix,
	}
}

func NewIdentityMatrix(n int) *Matrix {
	matrix := make([][]float64, n)
	for i := 0; i < n; i++ {
		matrix[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		matrix[i][i] = 1
	}
	return &Matrix{
		Arr: matrix,
	}
}
