/*
 fa_test.go

 Tests types implementing the FunctionApproximator interface.

 author: Timothy A. Mann
 date: August 28, 2014
*/
package goml

import (
	mat "github.com/skelterjohn/go.matrix"
	"math"
	"math/rand"
	"testing"
)

const (
	LINEAR_NOISE = 0.1
)

/*
LinearFitAndReturnMSE is used to test linear FunctionApproximators. It
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
func LinearFitAndReturnMSE(fa FunctionApproximator, t *testing.T) float64 {
	n := 100
	x := mat.Zeros(n, 1)
	y := mat.Zeros(n, 1)
	for i := 0; i < n; i++ {
		fi := float64(i)
		fn := float64(n)
		fx := fi / fn
		fy := (0.25*fx - 0.5) + rand.NormFloat64()*LINEAR_NOISE
		x.Set(i, 0, fx)
		y.Set(i, 0, fy)
	}
	err := fa.Fit(x, y)
	if err != nil {
		t.Error(err)
	}

	// Test bulk prediction
	yhatM, err := fa.PredictM(x)
	if err != nil {
		t.Error(err)
	}

	sqErrM := 0.0
	diffM, err := y.Minus(yhatM)
	if err != nil {
		t.Error(err)
	}

	sqErr := 0.0
	for i := 0; i < n; i++ {
		sqErrM += diffM.Get(i, 0) * diffM.Get(i, 0)

		v, err := fa.Predict(x.GetRowVector(i))
		if err != nil {
			t.Error(err)
		}
		diff := y.Get(i, 0) - v
		sqErr += diff * diff
	}

	mse := math.Max(sqErr, sqErrM) / float64(n)
	return mse
}

type Tanh struct{}

func (self Tanh) Eval(x float64) float64 {
	return math.Tanh(x)
}
func (self Tanh) Deriv(x float64) float64 {
	// 1 - Math.pow(Math.tanh(x),2)
	return 1 - math.Pow(math.Tanh(x), 2)
}

/*
TestSGD creates an SGD instance and tests whether it can fit a linear function.
*/
func TestSGD(t *testing.T) {
	lambda := 0.01
	numIterations := 1000
	learningRate := 0.1
	afunc := new(Tanh)
	sgd, err := NewSGD(L2_PENALTY, lambda, numIterations, learningRate, afunc)
	if err != nil {
		t.Error("Error while constructing SGD instance with Tanh activation function.", err)
	}
	mse := LinearFitAndReturnMSE(sgd, t)
	t.Log(sgd.Weights())
	if mse > LINEAR_NOISE {
		t.Error("MSE (", mse, ") is too large (w/Tanh activation function).")
	}

	sgd, err = NewSGD(L2_PENALTY, lambda, numIterations, learningRate, nil)
	if err != nil {
		t.Error("Error while constructing SGD instance without activation function.", err)
	}
	mse = LinearFitAndReturnMSE(sgd, t)
	t.Log(sgd.Weights())
	if mse > LINEAR_NOISE {
		t.Error("MSE (", mse, ") is too large (w/nil activation function).")
	}
}
