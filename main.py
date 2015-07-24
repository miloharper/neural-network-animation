import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot, animation
from neural_network import NeuralNetwork
from formulae import calculate_average_error
import parameters


class TrainingExample():
    def __init__(self, inputs, output):
        self.inputs = inputs
        self.output = output

if __name__ == "__main__":

    # Assemble neural network
    network = NeuralNetwork()
    network.add_layer(3)
    network.add_layer(4)
    network.add_layer(1)

    # Training set
    examples = [TrainingExample([0, 0, 1], 0),
                TrainingExample([0, 1, 1], 1),
                TrainingExample([1, 0, 1], 1),
                TrainingExample([0, 1, 0], 1),
                TrainingExample([1, 0, 0], 1),
                TrainingExample([1, 1, 1], 0),
                TrainingExample([0, 0, 0], 0)]

    # Generate a video of the neural network learning
    print "Generating a video of the neural network learning..."
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=parameters.frames_per_second, metadata=parameters.metadata)
    fig = pyplot.figure()
    with writer.saving(fig, parameters.file_name, 100):
        cumulative_error = None
        for i in xrange(parameters.training_iterations):
            average_error = calculate_average_error(cumulative_error, len(examples))
            cumulative_error = 0
            for e, example in enumerate(examples):
                cumulative_error += network.train(example)
                if i % parameters.iterations_per_frame == 0 or i == 1:
                    pyplot.clf()
                    pyplot.xlim(0, parameters.width)
                    pyplot.ylim(0, parameters.height)
                    network.draw()
                    pyplot.text(1, parameters.height - 1, "Iteration " + str(i + 1), fontsize=12)
                    pyplot.text(1, parameters.height - 2, "Example " + str(e + 1), fontsize=12)
                    if average_error:
                        pyplot.text(1, parameters.height - 3, "Average Error " + str(average_error) + "%", fontsize=12)
                    writer.grab_frame()
    print "Success! Open the file " + parameters.file_name + " to view the video."

    # Consider a new situation
    new_situation = [1, 1, 0]
    print "Considering a new situation " + str(new_situation) + "?"
    print network.think(new_situation)
