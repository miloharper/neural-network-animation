from neural_network import NeuralNetwork
from formulae import calculate_average_error, seed_random_number_generator
from video import generate_writer, annotate_frame, take_still
import parameters


class TrainingExample():
    def __init__(self, inputs, output):
        self.inputs = inputs
        self.output = output

if __name__ == "__main__":

    # Seed the random number generator
    seed_random_number_generator()

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

    # Create a video and image writer
    fig, writer = generate_writer()

    # Generate an image of the neural network before training
    print "Generating an image of the neural network before"
    network.do_not_think()
    network.draw()
    take_still("neural_network_before.png")

    # Generate a video of the neural network learning
    print "Generating a video of the neural network learning."
    print "There will be " + str(parameters.training_iterations * len(examples) / parameters.iterations_per_frame) + " frames."
    print "This may take a long time. Please wait..."
    with writer.saving(fig, parameters.video_file_name, 100):
        for i in xrange(parameters.training_iterations):
            cumulative_error = 0
            for e, example in enumerate(examples):
                cumulative_error += network.train(example)
                if i % parameters.iterations_per_frame == 1:
                    network.draw()
                    annotate_frame(i, e, average_error, example)
                    writer.grab_frame()
            average_error = calculate_average_error(cumulative_error, len(examples))
    print "Success! Open the file " + parameters.video_file_name + " to view the video."

    # Generate an image of the neural network after training
    print "Generating an image of the neural network after"
    network.do_not_think()
    network.draw()
    take_still("neural_network_after.png")

    # Consider a new situation
    new_situation = [1, 1, 0]
    print "Considering a new situation " + str(new_situation) + "?"
    print network.think(new_situation)
    network.draw()
    take_still("neural_network_new_situation.png")

