from neural_network import NeuralNetwork
from formulae import calculate_average_error
from video import generate_writer, new_frame, annotate_frame
import parameters


class TrainingExample():
    def __init__(self, inputs, output):
        self.inputs = inputs
        self.output = output

if __name__ == "__main__":

    # Assemble a neural network, with 3 neurons in the first layer
    # 4 neurons in the second layer and 1 neuron in the third layer
    network = NeuralNetwork([3, 4, 1])

    # Training set
    examples = [TrainingExample([0, 0, 1], 0),
                TrainingExample([0, 1, 1], 1),
                TrainingExample([1, 0, 1], 1),
                TrainingExample([0, 1, 0], 1),
                TrainingExample([1, 0, 0], 1),
                TrainingExample([1, 1, 1], 0),
                TrainingExample([0, 0, 0], 0)]

    # Generate a video of the neural network learning
    print "Generating a video of the neural network learning."
    print "There will be " + str(parameters.training_iterations * len(examples) / parameters.iterations_per_frame) + " frames."
    print "This may take a long time. Please wait..."
    fig, writer = generate_writer()
    with writer.saving(fig, parameters.file_name, 100):
        for i in xrange(parameters.training_iterations):
            cumulative_error = 0
            for e, example in enumerate(examples):
                cumulative_error += network.train(example)
                if i % parameters.iterations_per_frame == 1:
                    new_frame()
                    network.draw()
                    annotate_frame(i, e, average_error, example)
                    writer.grab_frame()
            average_error = calculate_average_error(cumulative_error, len(examples))
    print "Success! Open the file " + parameters.file_name + " to view the video."

    # Consider a new situation
    new_situation = [1, 1, 0]
    print "Considering a new situation " + str(new_situation) + "?"
    print network.think(new_situation)
