import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot
from math import fabs
from formulae import sigmoid, sigmoid_derivative, random_weight, get_color, adjust_line_to_perimeter_of_circle, layer_left_margin
import parameters


class Synapse():
    def __init__(self, input_neuron_index, x1, x2, y1, y2):
        self.input_neuron_index = input_neuron_index
        self.weight = random_weight()
        x1, x2, y1, y2 = adjust_line_to_perimeter_of_circle(x1, x2, y1, y2)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def draw(self):
        line = pyplot.Line2D((self.x1, self.x2), (self.y1, self.y2), lw=fabs(self.weight))
        pyplot.gca().add_line(line)


class Neuron():
    def __init__(self, x, y, previous_layer):
        self.x = x
        self.y = y
        self.output = 0
        self.synapses = []
        self.error = 0
        index = 0
        if previous_layer:
            for input_neuron in previous_layer.neurons:
                synapse = Synapse(index, x, input_neuron.x, y, input_neuron.y)
                self.synapses.append(synapse)
                index += 1

    def train(self, previous_layer):
        for synapse in self.synapses:
            # Find the strength of the input signal from the neuron in the layer below
            synapse_input = previous_layer.neurons[synapse.input_neuron_index].output
            # Propagate the error back down the synapse to the neuron in the layer below
            previous_layer.neurons[synapse.input_neuron_index].error += self.error * sigmoid_derivative(self.output) * synapse.weight
            # Adjust the synapse weight
            synapse.weight += synapse_input * self.error * sigmoid_derivative(self.output)
        return previous_layer

    def think(self, previous_layer):
        activity = 0
        for synapse in self.synapses:
            activity += synapse.weight * previous_layer.neurons[synapse.input_neuron_index].output
        self.output = sigmoid(activity)

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=parameters.neuron_radius, fill=True, color=get_color(self.output))
        pyplot.gca().add_patch(circle)
        for synapse in self.synapses:
            synapse.draw()


class Layer():
    def __init__(self, network, number_of_neurons):
        if len(network.layers) > 0:
            self.is_input_layer = False
            self.previous_layer = network.layers[-1]
            self.y = self.previous_layer.y + parameters.vertical_distance_between_layers
        else:
            self.is_input_layer = True
            self.previous_layer = None
            self.y = parameters.bottom_margin
        self.neurons = []
        x = layer_left_margin(number_of_neurons)
        for iteration in xrange(number_of_neurons):
            neuron = Neuron(x, self.y, self.previous_layer)
            self.neurons.append(neuron)
            x += parameters.horizontal_distance_between_neurons

    def think(self):
        for neuron in self.neurons:
            neuron.think(self.previous_layer)

    def draw(self):
        for neuron in self.neurons:
            neuron.draw()


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons):
        layer = Layer(self, number_of_neurons)
        self.layers.append(layer)

    def train(self, example):
        error = example.output - self.think(example.inputs)
        self.reset_errors()
        self.layers[-1].neurons[0].error = error
        for l in range(len(self.layers) - 1, 0, -1):
            for neuron in self.layers[l].neurons:
                self.layers[l - 1] = neuron.train(self.layers[l - 1])
        return fabs(error)

    def think(self, inputs):
        for layer in self.layers:
            if layer.is_input_layer:
                for index, value in enumerate(inputs):
                    self.layers[0].neurons[index].output = value
            else:
                layer.think()
        return self.layers[-1].neurons[0].output

    def draw(self):
        for layer in self.layers:
            layer.draw()

    def reset_errors(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.error = 0