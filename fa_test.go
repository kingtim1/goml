/*
 fa_test.go

 Tests types implementing the FunctionApproximator interface.

 author: Timothy A. Mann
 date: August 28, 2014
*/
package goml

import (
	mat "github.com/skelterjohn/go.matrix"
	"math/rand"
	"testing"
)

const (
	LINEAR_NOISE = 0.1
)

/*
LinearFitAndReturnError is used to test linear FunctionApproximators. It
generates a collection of samples according to a 1-d linear function, fits the
function approximator with the training samples, and then returns the Mean
Squared Error (MSE). The training set is purposefully equal to the test set so
that we can test that the passed function approximator learns the training data.

The targets of the linear function are perturbed by a small amount of Gaussian
noise.

Input
=====
fa : a function approximator
t : a testing instance

Returns
=======
the MSE of the fa on samples from the linear function
*/
func LinearFitAndReturnError(fa FunctionApproximator, t *testing.T) float64 {
	n := 100
	x := mat.Zeros(n, 1)
	y := mat.Zeros(n, 1)
	for i := 0; i < n; i++ {
		fi := float64(i)
		fn := float64(n)
		fx := fi / fn
		fy := (2.0*fx - 4.5) + rand.NormFloat64()*LINEAR_NOISE
		x.Set(i, 0, fx)
		y.Set(i, 0, fy)
	}
	err := fa.Fit(x, y)
	if err != nil {
		t.Error(err)
	}
	//t.Log("(Rows=", x.Rows(), "Cols=", x.Cols(), ")")

	sqErr := 0.0
	for i := 0; i < n; i++ {
		v, err := fa.Predict(x.GetRowVector(i))
		if err != nil {
			//t.Error(err)
		}
		diff := y.Get(i, 0) - v
		sqErr += diff * diff
	}

	return sqErr / float64(n)
}

/*
TestSGD creates an SGD instance and tests whether it can fit a linear function.
*/
func TestSGD(t *testing.T) {
	sgd, err := NewSGD(L2_PENALTY, 0.0, 1000, 0.1)
	if err != nil {
		t.Error("Error while constructing SGD instance.", err)
	}
	mse := LinearFitAndReturnError(sgd, t)
	t.Log(sgd.Weights())
	if mse > LINEAR_NOISE {
		t.Error("MSE (", mse, ") is too large.")
	}
}
