import parameters
from matplotlib import pyplot, animation


def generate_writer():
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=parameters.frames_per_second, metadata=parameters.metadata)
    fig = pyplot.figure()
    return fig, writer


def new_frame():
    pyplot.clf()
    pyplot.xlim(0, parameters.width)
    pyplot.ylim(0, parameters.height)


def annotate_frame(i, e, average_error):
    pyplot.text(1, parameters.height - 1, "Iteration " + str(i + 1), fontsize=12)
    pyplot.text(1, parameters.height - 2, "Example " + str(e + 1), fontsize=12)
    if average_error:
        pyplot.text(7, parameters.height - 1, "Average Error " + str(average_error) + "%", fontsize=12)
        error_bar(average_error)


def error_bar(average_error):
    border = pyplot.Rectangle((7, parameters.height - 3), 10, 1, color=(0, 0, 0), fill=False)
    pyplot.gca().add_patch(border)
    rectangle = pyplot.Rectangle((7, parameters.height - 3), 10 * average_error / 100, 1, color=(1, 0, 0))
    pyplot.gca().add_patch(rectangle)