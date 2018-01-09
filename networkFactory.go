package goNN

import (
	"strconv"
)


type NetworkConfig struct {
	Weights        [][][]float64
	InputIndicies  []int
	OutputIndicies []int
}

type NeuronConfig struct {
	Weight float64
	Recursive bool
	Bias bool

}

func FromMatrix(config NetworkConfig) *Network {
	net := NewNetwork()

	// each row/col indicates a new neuron.
	for i := range config.Weights[0] {
		net.neurons[i] = net.CreateNeuron(Identity)
	}

	// create the input channels
	for _, neuronIndex := range config.InputIndicies {
		in := makeConnectionChannel(false)
		net.Inputs = append(net.Inputs, in)
		net.neurons[neuronIndex].AddInput(1, in)
	}

	// create the output channels
	for _, neuronIndex := range config.OutputIndicies {
		out := makeConnectionChannel(false)
		net.Outputs = append(net.Outputs, out)
		net.neurons[neuronIndex].AddOutput(out)
	}

	net.initWeights(config.Weights[0], false)
	net.initWeights(config.Weights[1], true)

	return net
}

// NewFullyConnected takes an array of ints where each int is the size of the layer at that index.
func NewFullyConnected(layerSizes []int) *Network {
	net := NewNetwork()

	var lastLayer, layer []*Neuron

	for layerIndex, layerSize := range layerSizes {
		lastLayer = layer

		layer = make([]*Neuron, 0, layerSize)

		for neuronIndex := 0; neuronIndex < layerSize; neuronIndex++ {
			neuron := net.CreateNeuron(Identity)
			layer = append(layer, neuron)

			if layerIndex == 0 {
				c := makeConnectionChannel(false)

				// Add the connection as a input to the neuron
				neuron.AddInput(1, c)
				// register the connection as a network input too.
				net.Inputs = append(net.Inputs, c)
			} else {

				if layerIndex == len(layerSizes)-1 {
					c := makeConnectionChannel(false)

					// Add the connection as an output from the neuron
					neuron.AddOutput(c)

					// Register the connection as a network output too.
					net.Outputs = append(net.Outputs, c)
				}

				neuron.AddInput(0, net.biasPump)

				// Connect each of the neurons in the previous layer to the new neuron.
				for _, lastNeuron := range lastLayer {
					c := makeConnectionChannel(false)
					neuron.AddInput(1, lastNeuron.AddOutput(c))
				}
			}

		}
	}

	return net
}

func (net *Network) CreateNeuron(f ActivationFunction) *Neuron {
	id := len(net.neurons)
	n := NewNeuron(strconv.Itoa(id), f)
	net.neurons[id] = n
	return n
}

func (net *Network) initWeights(weights [][]float64, isReccurent bool) {
	for i, row := range weights {
		var from, to *Neuron
		from = net.neurons[i]

		for j, weight := range row {
			if weight != 0 {
				to = net.neurons[j]
				connect(from, to, weight, isReccurent)
			}
		}
	}
}
