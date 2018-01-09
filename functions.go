package goNN

import "math"

// ActivationFunction used by Neurons
type ActivationFunction func(float64) float64

// Identity is f(x)=x
func Identity(x float64) float64 { return x }

// Sigmoid activation function
func Sigmoid(x float64) float64 { return 1 / (1 + math.Exp(-x)) }

func Tanh(x float64) float64 {return math.Tanh(x)}

func ReLu(x float64) float64 {return math.Max(x,0)}
