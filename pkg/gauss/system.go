package gauss

import (
	"errors"
	"log"
	"math"
	"sync"
)

type System struct {
	M *Matrix   // указатель на матрицу
	B []float64 // вектор правой части
}

// ConditionNumber возвращает число обусловленности матрицы
func (s *System) ConditionNumber() (cond float64) {
	cs := s.Copy()
	inverse := cs.Inverse()
	return s.M.Norm() * inverse.Norm()
}

// Copy служит для глубокого копирования содержимого системы
func (s *System) Copy() (c *System) {
	b := make([]float64, len(s.B))
	copy(b, s.B)
	c = &System{
		M: s.M.Copy(),
		B: b,
	}
	return
}

// Det Функция для отыскания определителя
func (s *System) Det() (det float64, err error) {
	det = 1
	flag := s.forwardElim(true)
	for i := 0; i < len(s.M.Arr); i++ {
		det *= s.M.Arr[i][i]
	}
	det *= float64(flag)
	return
}

// Inverse служит для отыскания обратной матрицы
func (s *System) Inverse() (inverse *Matrix) {
	// конструируем единичную матрицу
	identity := NewIdentityMatrix(len(s.M.Arr))
	// сливаем её с исходной
	s.M.Extend(identity)
	// запускаем прямой ход
	s.forwardElim(false)
	// берем подматрицу исходной матрицы
	return s.M.Submatrix()
}

// GaussSolve функция-драйвер для решения системы методом Гаусса
func (s *System) GaussSolve() (x []float64, err error) {
	flag := s.forwardElim(false)
	if flag != -1 {
		log.Println("Вырожденная матрица")
		if s.M.Arr[flag][len(s.M.Arr[flag])] != 0 {
			log.Println("Неполна")
			return nil, errors.New("Inconsistent system")
		} else {
			log.Println("Бесконечное число решений")
			return nil, errors.New("Infinite number of solutions")
		}
	}
	x = s.backSub()
	return x, nil
}

// forwardElim работает в 2 режимах:
// в режиме определителя и в режиме прямого хода
// поведение опр. значением параметра det
// в первом случае flag содержит знак определителя
// во втором случае flag содержит либо номер вырожденной строки, либо значение -1 в случае успеха
func (s *System) forwardElim(det bool) (flag int) {
	// flag - служебное значение
	if det {
		flag = 1
	} else {
		flag = -1
	}
	var wg sync.WaitGroup
	wg.Add(len(s.M.Arr))
	for k := 0; k < len(s.M.Arr); k++ {
		// установка значений для поиска макс значения
		i_max := k
		v_max := s.M.Arr[i_max][k]

		// ищем самый большой по модулю элемент
		for i := k + 1; i < len(s.M.Arr); i++ {
			if math.Abs(s.M.Arr[i][k]) > v_max {
				v_max, i_max = s.M.Arr[i][k], i
			}
		}
		// избегаем деления на 0 т.е. сингулярных матриц
		if s.M.Arr[k][i_max] == 0 {
			return k
		}
		// делаем обмен
		if i_max != k {
			s.M.Arr[k], s.M.Arr[i_max] = s.M.Arr[i_max], s.M.Arr[k]
			if !det {
				s.B[k], s.B[i_max] = s.B[i_max], s.B[k]
			}
			if det {
				flag *= -1
			}
		}

		// идем циклом по строкам ниже текущей, производя вычитание
		for i := k + 1; i < len(s.M.Arr); i++ {
			factor := s.M.Arr[i][k] / s.M.Arr[k][k]
			for j := k + 1; j < len(s.M.Arr); j++ {
				s.M.Arr[i][j] -= s.M.Arr[k][j] * factor
			}
			if !det {
				s.B[i] -= s.B[k] * factor
			}
			s.M.Arr[i][k] = 0
		}
	}
	// матрица успешно приведена
	return flag
}

// forwardElimParallel выполняет частично параллельный прямой ход работает в 2 режимах:
// в режиме определителя и в режиме прямого хода
// поведение опр. значением параметра det
// в первом случае flag содержит знак определителя
// во втором случае flag содержит либо номер вырожденной строки, либо значение -1 в случае успеха
func (s *System) forwardElimParallel(det bool) (flag int) {
	// flag - служебное значение
	if det {
		flag = 1
	} else {
		flag = -1
	}
	var wg sync.WaitGroup
	wg.Add(len(s.M.Arr))
	for k := 0; k < len(s.M.Arr); k++ {
		// установка значений для поиска макс значения
		i_max := k
		v_max := s.M.Arr[i_max][k]

		// ищем самый большой по модулю элемент
		for i := k + 1; i < len(s.M.Arr); i++ {
			if math.Abs(s.M.Arr[i][k]) > v_max {
				v_max, i_max = s.M.Arr[i][k], i
			}
		}
		// избегаем деления на 0 т.е. сингулярных матриц
		if s.M.Arr[k][i_max] == 0 {
			return k
		}
		// делаем обмен
		if i_max != k {
			s.M.Arr[k], s.M.Arr[i_max] = s.M.Arr[i_max], s.M.Arr[k]
			if !det {
				s.B[k], s.B[i_max] = s.B[i_max], s.B[k]
			}
			if det {
				flag *= -1
			}
		}

		if !det {
			for i := k + 1; i < len(s.M.Arr); i++ {
				factor := s.M.Arr[i][k] / s.M.Arr[k][k]
				for j := k + 1; j < len(s.M.Arr); j++ {
					s.M.Arr[i][j] -= s.M.Arr[k][j] * factor
				}
				s.B[i] -= s.B[k] * factor
				s.M.Arr[i][k] = 0
			}
		}
		// идем циклом по строкам ниже текущей, производя вычитание
		var wg sync.WaitGroup        // создаем группу ожидания из горутин
		wg.Add(len(s.M.Arr) - k - 1) // обозначим количество горутин в группе
		for i := k + 1; i < len(s.M.Arr); i++ {
			i := i
			// запускаем функцию в горутину
			go func() {
				factor := s.M.Arr[i][k] / s.M.Arr[k][k]
				for j := k + 1; j < len(s.M.Arr); j++ {
					s.M.Arr[i][j] -= s.M.Arr[k][j] * factor
				}
				if !det {
					s.B[i] -= s.B[k] * factor
				}
				s.M.Arr[i][k] = 0
				// горутина сигнализирует группе о своем завершении
				wg.Done()
			}()
		}
		// ждем успешного завершения группы
		wg.Wait()
	}
	// матрица успешно приведена
	return flag
}

// backSub выполняет обратный ход
func (s *System) backSub() (x []float64) {
	N := len(s.M.Arr)
	x = make([]float64, N)
	for i := N - 1; i >= 0; i-- {
		x[i] = s.B[i]
		for j := i + 1; j < N; j++ {
			x[i] -= s.M.Arr[i][j] * x[j]
		}
		x[i] = x[i] / s.M.Arr[i][i]
	}
	return
}

func NewSystem(m *Matrix, b []float64) *System {
	return &System{
		M: m,
		B: b,
	}
}
