package gauss

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMatrix_Extend(t *testing.T) {
	type fields struct {
		Arr [][]float64
	}
	type args struct {
		another *Matrix
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		{"simple",
			fields{
				Arr: [][]float64{
					{1, 0, 0},
					{0, 1, 0},
					{0, 0, 1},
				},
			},
			args{
				another: NewMatrixFromArr([][]float64{
					{1, 0, 0},
					{0, 1, 0},
					{0, 0, 1},
				}),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Matrix{
				Arr: tt.fields.Arr,
			}
			m.Extend(tt.args.another)
			want := NewMatrixFromArr([][]float64{
				{1, 0, 0, 1, 0, 0},
				{0, 1, 0, 0, 1, 0},
				{0, 0, 1, 0, 0, 1},
			})
			assert.True(t, m.Equal(want))
		})
	}
}

func TestMatrix_Submatrix(t *testing.T) {
	type fields struct {
		Arr [][]float64
	}
	matrices := [][][]float64{
		{
			{1, 0, 0, 1, 0, 0},
			{0, 1, 0, 0, 1, 0},
			{0, 0, 1, 0, 0, 1},
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
		name    string
		fields  fields
		wantNew *Matrix
	}{
		{
			"simple",
			fields{
				matrices[0],
			},
			NewMatrixFromArr(answers[0]),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Matrix{
				Arr: tt.fields.Arr,
			}
			if gotNew := m.Submatrix(); !reflect.DeepEqual(gotNew, tt.wantNew) {
				t.Errorf("Submatrix() = %v, want %v", gotNew, tt.wantNew)
			}
		})
	}
}
