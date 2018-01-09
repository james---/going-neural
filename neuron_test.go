package goNN

import "testing"

func TestCreateInput(t *testing.T) {
	neuron := NewNeuron("1", Identity)
	numInputs := len(neuron.inputs)
	if numInputs != 0 {
		t.Errorf("Expected neuron to have 0 inputs before neuron.createInput but was %d", numInputs)
	}

	neuron.AddInput(1, make(chan float64))
	numInputs = len(neuron.inputs)
	if numInputs != 1 {
		t.Errorf("Expected neuron to have 1 input after neuron.createInput but was %d", numInputs)
	}
}

func TestNeuronInputOutput(t *testing.T) {
	neuron := NewNeuron("1", Identity)
	in := neuron.AddInput(1, makeConnectionChannel(false))
	out := neuron.AddOutput(makeConnectionChannel(false))
	stop := make(chan struct{}, 0)
	neuron.start(stop)

	in <- 0.5

	if 0.5 != <-out {
		t.Errorf("Incorrect output from neuron")
	}

	close(stop)
}

func TestNeuron(t *testing.T) {
	neuron := NewNeuron("1", func(x float64) float64 { return x * 2 })
	in1 := neuron.AddInput(1, makeConnectionChannel(false))
	in2 := neuron.AddInput(1, makeConnectionChannel(false))
	in3 := neuron.AddInput(1, makeConnectionChannel(false))
	out := neuron.AddOutput(makeConnectionChannel(false))

	stop := make(chan struct{}, 0)
	neuron.start(stop)

	in1 <- 1
	in2 <- 2
	in3 <- 3

	result := <-out
	if 12 != result {
		t.Errorf("Expected output of 12.0 but was %f", result)
	}

	close(stop)
}

func TestNeuronCanConnectToOthers(t *testing.T) {
	neuron1 := NewNeuron("1", Identity)
	neuron2 := NewNeuron("2", Identity)
	in1 := neuron1.AddInput(1, makeConnectionChannel(false))
	in2 := neuron1.AddInput(1, makeConnectionChannel(false))
	neuron2.AddInput(0.5, neuron1.AddOutput(makeConnectionChannel(false)))
	out := neuron2.AddOutput(makeConnectionChannel(false))

	stop := make(chan struct{}, 0)
	neuron1.start(stop)
	neuron2.start(stop)

	in1 <- 1
	in2 <- 2

	result := <-out
	if 1.5 != result {
		t.Errorf("Expected output of 1.5 but was %f", result)
	}

	close(stop)
}
