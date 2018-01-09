package goNN

import "log"

// The Network has a number of inputs and outputs
type Network struct {
	// The channels that feed inputs into the network
	Inputs []chan<- float64

	// The channels that provide the output of the network
	Outputs []<-chan float64

	neurons  map[int]*Neuron
	stop     chan struct{}
	biasPump chan float64
}

// NewNetwork creates a new empty Netwrok
func NewNetwork() *Network {
	return &Network{
		Inputs:   make([]chan<- float64, 0, 1),
		Outputs:  make([]<-chan float64, 0, 1),
		neurons:  make(map[int]*Neuron),
		biasPump: make(chan float64, 10)}
}

func (net *Network) Start() {
	net.stop = make(chan struct{}, 0)

	// Start pushing 1s into the bias pump
	go func() {
		for {
			select {
			case net.biasPump <- 1:
			case <-net.stop:
				return
			}
		}
	}()

	// Start the individual neurons
	for _, n := range net.neurons {
		n.start(net.stop)
	}

}

func (net *Network) Stop() {
	close(net.stop)
}

// Input passes the given sense inputs to the individual which will 'think' and
// then provide it's output response via the returned channel
func (net *Network) Input(is []float64) []<-chan float64 {
	if len(is) != len(net.Inputs) {
		if len(is) > len(net.Inputs) {
			log.Panicf("Too many inputs to network. Expected %d, got %d", len(net.Inputs), len(is))
		} else {
			log.Panicf("Too few inputs to network. Expected %d, got %d", len(net.Inputs), len(is))
		}
	}

	go func() {
		for index, in := range is {
			net.Inputs[index] <- in
		}
	}()
	return net.Outputs
}

func connect(from *Neuron, to *Neuron, weight float64, isRecurrent bool) {
	connectingChan := makeConnectionChannel(isRecurrent)
	from.AddOutput(connectingChan)
	to.AddInput(weight, connectingChan)
}

// makeConnectionChannel creates a channel
func makeConnectionChannel(recurrent bool) chan float64 {
	var c chan float64
	if recurrent {
		c = make(chan float64, 1)
		// Filling the single slot with a 0 will mean its always passing the value from t-1, ie recurrent.
		c <- 0
		return c
	}
	return make(chan float64)
}
