/*
 sgd.go

 Implementation of Stochastic Gradient Descent (SGD).

 author: Timothy A. Mann
 date: August 28, 2014
*/

package goml

import (
	"fmt"
	mat "github.com/skelterjohn/go.matrix"
	"math/rand"
)

const (
	L1_PENALTY = 0
	L2_PENALTY = 1
)

/*
SGD performs Stochastic Gradient Descent.
*/
type SGD struct {
	/*
		The penalty type to apply (L1_PENALTY or L2_PENALTY).
	*/
	PenaltyType int
	/*
		The regularization parameter.
	*/
	Lambda float64
	/*
		The number of iterations to train for.
	*/
	NumIterations int
	/*
		The learning rate used during training.
	*/
	LearningRate float64

	/*
		The number of dimensions of a valid input vector.
	*/
	inputDims int

	/*
		A LinearFunction.
	*/
	f *LinearFunction
}

/*
NewSGD constructs a new SGD instance.

Input
=====
penaltyType : the type of regularization penalty to user during fitting (either L1_PENALTY or L2_PENALTY)
lambda : the regularization parameter (should >= 0)
numIterations : the number of iterations to run during fitting. One iteration corresponds to updating with a single sample.
learningRate : the constant learning rate parameter to use during training

Returns
=======
a pointer to a new (untrained) SGD instance or an error
*/
func NewSGD(penaltyType int, lambda float64, numIterations int, learningRate float64) (*SGD, error) {
	var f *SGD = new(SGD)
	if penaltyType != L1_PENALTY && penaltyType != L2_PENALTY {
		return nil, fmt.Errorf("Invalid regularization penalty type. Valid types are L1_PENALTY or L2_PENALTY.")
	}
	f.PenaltyType = penaltyType
	if lambda < 0.0 {
		return nil, fmt.Errorf("Regularization parameter cannot be negative.")
	}
	f.Lambda = lambda
	if numIterations < 1 {
		return nil, fmt.Errorf("numIterations must be positive.")
	}
	f.NumIterations = numIterations
	f.LearningRate = learningRate

	f.inputDims = 0
	f.f = nil
	return f, nil
}

/*
NewCopy returns a new instance of *SGD with the same parameters as this instance,
but it does not copy any weights. So the new instance can be trained with any
(x, y) training samples.

Returns
=======
a new instance of *SGD with the same parameters as this instance
*/
func (self *SGD) NewCopy() (*SGD, error) {
	return NewSGD(self.PenaltyType, self.Lambda, self.NumIterations, self.LearningRate)
}

func (self *SGD) Fit(x mat.MatrixRO, y mat.MatrixRO) error {
	if x.Rows() != y.Rows() {
		return fmt.Errorf("The number of rows in x (%d) does not match the number of rows in y (%d). The matrix x should contain one input vector per row and the vector y should be a column vector containing labels for each input vector.", x.Rows(), y.Rows())
	}
	if y.Cols() != 1 {
		return fmt.Errorf("y must be a column vector.")
	}

	// The number of samples in the data set
	n := x.Rows()
	// Get a dense version of the input matrix
	dx := x.DenseMatrix()

	if self.f == nil {
		self.inputDims = x.Cols()
		self.f = new(LinearFunction)
		self.f.Weights = *mat.Zeros(self.inputDims+1, 1)
	} else if self.inputDims != x.Cols() {
		return fmt.Errorf("The number of columns in matrix x does not match the dimension of previous training data. Please construct a new SGD instance.")
	}

	for i := 0; i < self.NumIterations; i++ {
		index := rand.Intn(n)
		xrow := dx.GetRowVector(index)
		xrowb := self.addBiasToVector(xrow)
		yhat, err := self.f.Predict(xrowb)
		if err != nil {
			return fmt.Errorf("Error while predicting with internal linear model. %v", err)
		}

		diff := y.Get(index, 0) - yhat
		for j := 0; j < self.inputDims+1; j++ {
			// Get the old weight value
			oldw := self.f.Weights.Get(j, 0)
			// Calculate the gradient of the squared error
			grad := 0.0
			if j < self.inputDims {
				grad = (diff * -xrow.Get(0, j))
			} else {
				// Gradient for the bias
				grad = -diff
			}

			// Calculate the gradient of the regularization penalty
			gpen := 0.0
			if self.PenaltyType == L1_PENALTY {
				gpen = self.Lambda * signum(oldw)
			} else {
				gpen = self.Lambda * oldw
			}
			// Calculate the change in weight
			alpha := self.LearningRate / float64(self.inputDims)
			deltaw := alpha * (grad + gpen)
			neww := oldw - deltaw
			// Set the new weight
			self.f.Weights.Set(j, 0, neww)
		}
	}

	return nil
}

func (self SGD) addBiasToVector(x mat.MatrixRO) *mat.DenseMatrix {
	xb := mat.Ones(1, self.InputDims()+1)
	for i := 0; i < self.InputDims(); i++ {
		xb.Set(0, i, x.Get(0, i))
	}
	return xb
}

func (self SGD) Predict(x mat.MatrixRO) (float64, error) {
	if self.f != nil {
		if x.Cols() != self.InputDims() {
			return 0, fmt.Errorf("x has %d columns. Expected %d.", x.Cols(), self.InputDims())
		}
		// Add a bias to the input vector
		xb := self.addBiasToVector(x)
		// Make a prediction
		return self.f.Predict(xb)
	} else {
		return 0, fmt.Errorf("Cannot predict before running the Fit method.")
	}
}

func (self SGD) InputDims() int {
	return self.inputDims
}

/*
Weights returns the weight vector obtained during the fitting process. If Fit()
has not been executed yet, then the behavior of this method is undefined.

Returns
=======
a column vector of weights (including an additional weight for the bias)
*/
func (self SGD) Weights() mat.DenseMatrix {
	if self.f != nil {
		return self.f.Weights
	} else {
		return *mat.Zeros(1, 1)
	}
}

/*
SqError computes the squared difference between two scalar values a and b.

Input
=====
a : a scalar value
b : a scalar value

Returns
=======
the squared difference between a and b
*/
func SqError(a, b float64) float64 {
	diff := a - b
	return diff * diff
}

/*
signum returns the sign of a float64 as -1, 0, or 1.
*/
func signum(a float64) float64 {
	switch {
	case a < 0:
		return -1.0
	case a > 0:
		return +1.0
	}
	return 0.0
}
