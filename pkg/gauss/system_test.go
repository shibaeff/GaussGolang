package gauss

import (
	"math"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSystem_forwardElim(t *testing.T) {
	type fields struct {
		M *Matrix
		B []float64
	}
	matrices := [][][]float64{
		{
			{3.0, 2.0, -4.0},
			{2.0, 3.0, 3.0},
			{5.0, -3, 1.0},
		},
	}
	//ans := [][][]float64{
	//	{
	//		{1.0, 0, 0},
	//		{0, 1.0, 0},
	//		{0, 0, 1.0},
	//	},
	//}
	vectors := [][]float64{
		{3, 15, 14},
	}
	tests := []struct {
		name     string
		fields   fields
		wantFlag int
	}{
		{
			"simple",
			fields{
				NewMatrixFromArr(matrices[0]),
				vectors[0],
			},
			-1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &System{
				M: tt.fields.M,
				B: tt.fields.B,
			}
			if gotFlag := s.forwardElim(false); gotFlag != tt.wantFlag {
				t.Errorf("forwardElim() = %v, want %v", gotFlag, tt.wantFlag)
			}
			assert.True(t, tt.fields.M.IsLowerTriangular())
		})
	}
}

func TestSystem_GaussSolve(t *testing.T) {
	type fields struct {
		M *Matrix
		B []float64
	}
	matrices := [][][]float64{
		{
			{3.0, 2.0, -4.0},
			{2.0, 3.0, 3.0},
			{5.0, -3, 1.0},
		},
	}
	vectors := [][]float64{
		{3, 15, 14},
	}
	tests := []struct {
		name    string
		fields  fields
		wantErr bool
	}{
		{
			"simple",
			fields{
				M: NewMatrixFromArr(matrices[0]),
				B: vectors[0],
			},
			false,
		},
	}
	answers := [][]float64{
		{3, 1, 2},
	}
	for testNumber, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &System{
				M: tt.fields.M,
				B: tt.fields.B,
			}
			x, err := s.GaussSolve()
			if (err != nil) != tt.wantErr {
				t.Errorf("GaussSolve() error = %v, wantErr %v", err, tt.wantErr)
			}

			for i := 0; i < len(answers[testNumber]); i++ {
				assert.LessOrEqual(t, math.Abs(answers[testNumber][i]-x[i]), 0.0001)
			}
		})
	}
}

func TestSystem_Det(t *testing.T) {
	type fields struct {
		M *Matrix
		B []float64
	}
	matrices := [][][]float64{
		{
			{3.0, 2.0, -4.0},
			{2.0, 3.0, 3.0},
			{5.0, -3, 1.0},
		},
		{
			{0, 0, 0},
			{2.0, 3.0, 3.0},
			{5.0, -3, 1.0},
		},
	}
	answers := []float64{
		146, 0,
	}
	tests := []struct {
		name    string
		fields  fields
		wantDet float64
		wantErr bool
	}{
		{
			name: "simple",
			fields: fields{
				M: NewMatrixFromArr(matrices[0]),
				B: nil,
			},
			wantDet: answers[0],
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &System{
				M: tt.fields.M,
				B: tt.fields.B,
			}
			gotDet, err := s.Det()
			if (err != nil) != tt.wantErr {
				t.Errorf("Det() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			assert.LessOrEqual(t, math.Abs(gotDet-answers[0]), 0.0001)
		})
	}
}

func TestSystem_Inverse(t *testing.T) {
	type fields struct {
		M *Matrix
		B []float64
	}
	matrices := [][][]float64{
		{
			{1, 0, 0},
			{0, 1, 0},
			{0, 0, 1},
		},
	}
	answers := [][][]float64{
		{
			{1, 0, 0},
			{0, 1, 0},
			{0, 0, 1},
		},
	}
	tests := []struct {
		name        string
		fields      fields
		wantInverse *Matrix
	}{
		{
			"identity",
			fields{
				M: NewMatrixFromArr(matrices[0]),
				B: []float64{1, 1, 1},
			},
			NewMatrixFromArr(answers[0]),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &System{
				M: tt.fields.M,
				B: tt.fields.B,
			}
			if gotInverse := s.Inverse(); !reflect.DeepEqual(gotInverse, tt.wantInverse) {
				t.Errorf("Inverse() = %v, want %v", gotInverse, tt.wantInverse)
			}
		})
	}
}

func TestSystem_forwardElimParallel(t *testing.T) {
	type fields struct {
		M *Matrix
		B []float64
	}
	matrices := [][][]float64{
		{
			{3.0, 2.0, -4.0},
			{2.0, 3.0, 3.0},
			{5.0, -3, 1.0},
		},
	}
	//ans := [][][]float64{
	//	{
	//		{1.0, 0, 0},
	//		{0, 1.0, 0},
	//		{0, 0, 1.0},
	//	},
	//}
	vectors := [][]float64{
		{3, 15, 14},
	}
	tests := []struct {
		name     string
		fields   fields
		wantFlag int
	}{
		{
			"simple",
			fields{
				NewMatrixFromArr(matrices[0]),
				vectors[0],
			},
			-1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &System{
				M: tt.fields.M,
				B: tt.fields.B,
			}
			if gotFlag := s.forwardElimParallel(false); gotFlag != tt.wantFlag {
				t.Errorf("forwardElim() = %v, want %v", gotFlag, tt.wantFlag)
			}
			assert.True(t, tt.fields.M.IsLowerTriangular())
		})
	}
}
