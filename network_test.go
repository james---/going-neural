package goNN

import (
	"testing"
)

func TestNetwork(t *testing.T) {
	net := NewNetwork()

	out := net.CreateNeuron(Identity)
	oc := makeConnectionChannel(false)
	out.AddOutput(oc)
	net.Outputs = append(net.Outputs, oc)

	if len(net.Outputs) != 1 {
		t.Errorf("Expected 1 output but was %d", len(net.Outputs))
	}

	in1 := net.CreateNeuron(Identity)
	c1 := makeConnectionChannel(false)
	in1.AddInput(1, c1)

	in2 := net.CreateNeuron(Identity)
	c2 := makeConnectionChannel(false)
	in2.AddInput(1, c2)
	net.Inputs = append(net.Inputs, c1, c2)

	connect(in1, out, 1, false)
	connect(in2, out, 1, false)

	if len(net.Inputs) != 2 {
		t.Errorf("Expected 2 inputs but was %d", len(net.Inputs))
	}

	defer net.Stop()
	net.Start()

	result := <-net.Input([]float64{1, 2})[0]

	if result != 3 {
		t.Errorf("Expected output of 3 but was %f", result)
	}

}

func TestNetworkBuiltFromAdjacencyMatrix(t *testing.T) {

	// [1]
	//    \ w=0.5
	//     [3]
	//    / w=0.5
	// [2]

	// + |  1  |  2  |  3  |
	// ― +  ―  +  ―  +  ―  +
	// 1 |  0  |  0  | 0.5 |
	// ― +  ―  +  ―  +  ―  +
	// 2 |  0  |  0  | 0.5 |
	// ― +  ―  +  ―  +  ―  +
	// 3 |  0  |  0  |  0  |
	// ― +  ―  +  ―  +  ―  +

	// TODO Need to specify functions
	net := FromMatrix(
		NetworkConfig{
			Weights: [][][]float64{
				{{0, 0, 0.5},
					{0, 0, 0.5},
					{0.5, 0, 0}},
				{{0, 0, 0},
					{0, 0, 0},
					{0, 0, 0}}},
			InputIndicies:  []int{0, 1},
			OutputIndicies: []int{2},
		},
	)

	if len(net.Inputs) != 2 {
		t.Errorf("Expected 2 inputs but was %d", len(net.Inputs))
	}

	if len(net.Outputs) != 1 {
		t.Errorf("Expected 1 output but was %d", len(net.Outputs))
	}

	defer net.Stop()
	net.Start()

	net.Input([]float64{1, 3})

	out := <-net.Outputs[0]

	if out != 2 {
		t.Errorf("Expected output 2 but was %f", out)
	}
}

func TestRecurrentNetwork(t *testing.T) {

	// [1]-1.0-[2]--
	//        /   \
	//        \0.5/
	//				 --
	//
	// Forward Matrix
	// + |  1  |  2  |
	// ― +  ―  +  ―  +
	// 1 |  0  | 1.0 |
	// ― +  ―  +  ―  +
	// 2 |  0  |  0  |
	// ― +  ―  +  ―  +

	// Recurrent Matrix
	// + |  1  |  2  |
	// ― +  ―  +  ―  +
	// 1 |  0  |  0  |
	// ― +  ―  +  ―  +
	// 2 |  0  | 0.5 |
	// ― +  ―  +  ―  +

	// TODO Need to specify functions
	net := FromMatrix(
		NetworkConfig{
			Weights: [][][]float64{
				{
					// feed forward weights
					{0, 1},
					{0, 0}},
				{
					// recursive weights
					{0, 0},
					{0, 0.5}}},
			InputIndicies:  []int{0},
			OutputIndicies: []int{1},
		},
	)

	if len(net.Inputs) != 1 {
		t.Errorf("Expected 1 inputs but was %d", len(net.Inputs))
	}

	if len(net.Outputs) != 1 {
		t.Errorf("Expected 1 output but was %d", len(net.Outputs))
	}

	defer net.Stop()
	net.Start()

	net.Input([]float64{1})
	out := <-net.Outputs[0]

	if out != 1 {
		t.Errorf("Expected output #1 to be 1 but was %f", out)
	}

	net.Input([]float64{1})
	out = <-net.Outputs[0]

	if out != 1.5 {
		t.Errorf("Expected output #2 to be 1.5 but was %f", out)
	}

	net.Input([]float64{1})
	out = <-net.Outputs[0]

	if out != 1.75 {
		t.Errorf("Expected output #2 to be 1.75 but was %f", out)
	}
}

func TestFullyConnectedFeedForward(t *testing.T) {
	net := NewFullyConnected([]int{10, 5, 2, 1})
	net.Start()

	inputsValues := make([]float64, 10)
	for i, _ := range inputsValues {
		inputsValues[i] = float64(i)
	}
	output := <-net.Input(inputsValues)[0]
	if output != 450 {
		t.Errorf("Expected output to be 450 but was %f", output)
	}
}

func TestBiasPump(t *testing.T) {
	net := NewNetwork()
	net.Start()
	for i := 0; i < 5; i++ {
		b := <-net.biasPump
		if 1 != b {
			t.Errorf("Expected bias output to be 1 but was %f", b)
		}
	}
	net.Stop()
drain:
	for {
		select {
		case <-net.biasPump:
		default:
			break drain
		}
	}
}
