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
		Eval evalutes this function at the point specified by the given vector
		and returns a scalar value.

		Input
		=====
		instance : a row vector

		Returns
		=======
		a scalar value or an error
	*/
	Eval(instance mat.MatrixRO) (float64, error)

	/*
		InputDims returns the number of dimensions of a valid input vector.

		Returns
		=======
		the number of dimensions of a valid input vector
	*/
	InputDims() uint
}

/*
FunctionApproximator is a function that can be trained given a labeled set of
input vector, scalar value pairs.
*/
type FunctionApproximator interface {
	Function

	/*
		Fit fits this function approximator to the training data specified by
		the matrix x and the vector y.

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
	Weights mat.MatrixRO
}

func (f LinearFunction) Eval(instance mat.MatrixRO) (float64, error) {
	if uint(instance.Cols()) != f.InputDims() {
		return 0, fmt.Errorf("Instance has wrong dimensions.")
	}
	value, err := f.Weights.Times(instance)
	return value.Get(0, 0), err
}

func (f LinearFunction) InputDims() uint {
	return uint(f.Weights.Rows())
}
