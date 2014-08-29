/*
 goml.go

 A few machine learning algorithms implemented in go.

 author: Timothy A. Mann
 date: August 28, 2014
*/

package goml

import (
	"fmt"
	mat "github.com/skelterjohn/go.matrix"
)

/*
 Function is a mapping from a vector space to a float64.
*/
type Function interface {

	/*
		Predict evalutes this function at the point specified by the given
		vector and returns a scalar value.

		Input
		=====
		instance : a row vector

		Returns
		=======
		a scalar value or an error
	*/
	Predict(instance mat.MatrixRO) (float64, error)

	/*
		PredictM evaluates each row of the specified matrix.

		Input
		=====
		instances : a matrix where each row corresponds to an input vector

		Returns
		=======
		a vector containing one prediction for each row vector in instances
	*/
	PredictM(instances mat.MatrixRO) (mat.MatrixRO, error)

	/*
		InputDims returns the number of dimensions of a valid input vector.

		Returns
		=======
		the number of dimensions of a valid input vector
	*/
	InputDims() int
}

/*
FunctionApproximator is a function that can be trained given a labeled set of
input vector, scalar value pairs.
*/
type FunctionApproximator interface {
	Function

	/*
		Fit fits this function approximator to the training data specified by
		the matrix x and the vector y. Generally this method needs to be called
		before Predict() can be called.

		Input
		=====
		x : a matrix where each row is an input vector
		y : a column vector where each element corresponds to the desired output
		 for the corresponding row of x

		Returns
		=======
		an error if the fitting process fails
	*/
	Fit(x mat.MatrixRO, y mat.MatrixRO) error
}

/*
 LinearFunction is a Function that evaluates input vectors by multiplying them
 by a weight vector.
*/
type LinearFunction struct {
	Weights mat.DenseMatrix
	AFunc   ActivationFunction
}

func (f LinearFunction) Predict(x mat.MatrixRO) (float64, error) {
	if x.Cols() != f.InputDims() {
		return 0, fmt.Errorf("x has %d columns. Expected %d.", x.Cols(), f.InputDims())
	}
	value, err := x.Times(&f.Weights)
	if f.AFunc != nil {
		return f.AFunc.Eval(value.Get(0, 0)), err
	} else {
		return value.Get(0, 0), err
	}
}

func (f LinearFunction) PredictM(x mat.MatrixRO) (mat.MatrixRO, error) {
	if x.Cols() != f.InputDims() {
		return nil, fmt.Errorf("x has %d columns. Expected %d.", x.Cols(), f.InputDims())
	}
	y, err := x.Times(&f.Weights)
	if err != nil {
		return nil, fmt.Errorf("Error predicting before applying activation function. %v", err)
	}
	var yprime mat.MatrixRO = nil
	if f.AFunc != nil {
		yprime = Apply(y, f.AFunc.Eval)
	} else {
		yprime = y
	}
	return yprime, nil
}

/*
InputDims returns the number of dimensions of a valid input vector.

Returns
=======
the number of dimensions of a valid input vector
*/
func (f LinearFunction) InputDims() int {
	return f.Weights.Rows()
}

/*
ActivationFunction is used to apply a non-linear transformation to the output of
a linear function. Activation functions need to be differentiable so that they
can be used with gradient descent.
*/
type ActivationFunction interface {
	/*
		Eval computes the value of this activation function at x.

		Input
		=====
		x : a scalar value

		Returns
		=======
		the value of this activation function at x
	*/
	Eval(x float64) float64

	/*
		Deriv computes the derivative of this activation function at x.

		Input
		=====
		x : a scalar value

		Returns
		=======
		the derivative of this activation function at x
	*/
	Deriv(x float64) float64
}

/*
SFunction is a function that maps a scalar value to another scalar value.
*/
type SFunction func(float64) float64

/*
Apply applies the function f to each element in A and returns a new matrix with
the results.

Input
=====
A : a matrix
f : a function from scalar values to scalar values

Returns
=======
a matrix derived by applying f to each element in A. If f is nil, then this
function just returns A.
*/
func Apply(A mat.MatrixRO, f SFunction) mat.MatrixRO {
	if f == nil {
		return A
	}
	B := mat.Zeros(A.Rows(), A.Cols())
	for r := 0; r < A.Rows(); r++ {
		for c := 0; c < A.Cols(); c++ {
			x := A.Get(r, c)
			y := f(x)
			B.Set(r, c, y)
		}
	}
	return B
}
