package goNN

// Neuron represents the processing unit of a network
type Neuron struct {
	id string
	// The channels that the neuron will receive input signals from.
	inputs []<-chan float64
	// Slice of float64s representing the weights for each respective channel in the inputs Slice
	weights []float64
	// The channel that the neuron's output signal will be sent to.
	outputs []chan<- float64
	// The neurons activation function.
	function ActivationFunction
}

// NewNeuron will create and return a pointer to a new neuron
func NewNeuron(id string, f ActivationFunction) *Neuron {
	n := &Neuron{
		id:       id,
		inputs:   make([]<-chan float64, 0, 1),
		weights:  make([]float64, 0, 1),
		outputs:  make([]chan<- float64, 0, 1),
		function: f}

	return n
}

// AddInput creates, adds and returns a new input into the neuron.
func (n *Neuron) AddInput(w float64, c chan float64) chan float64 {
	n.inputs = append(n.inputs, c)
	n.weights = append(n.weights, w)
	return c
}

// AddOutput creates, adds and returns a new output into the neuron.
func (n *Neuron) AddOutput(o chan float64) chan float64 {
	n.outputs = append(n.outputs, o)
	return o
}

func (n *Neuron) start(stop <-chan struct{}) {
	//log.Printf("start %s ", n.id)
	go func() {

		for {
			var signal float64

			for i, inputConnection := range n.inputs {
				select {
				case signalInput := <-inputConnection:
					//log.Printf("(%s) %f = %f + (%f*%f)", n.id, signal+(n.weights[i]*signalInput), signal, n.weights[i], signalInput)
					signal = signal + (n.weights[i] * signalInput)
				case <-stop:
					//log.Printf("stop")
					return
				}
			}

			outputSignal := n.function(signal)
			// log.Printf("(%s) %f := n.function(%f)", n.id, outputSignal, signal)

			for _, outputConnection := range n.outputs {
				select {
				case outputConnection <- outputSignal:
				case <-stop:
					return
				}
			}
		}
	}()
}