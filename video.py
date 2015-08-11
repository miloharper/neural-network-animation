import parameters
from matplotlib import pyplot, animation


def generate_writer():
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=parameters.frames_per_second, metadata=parameters.metadata)
    fig = pyplot.figure()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    pyplot.xlim(0, parameters.width)
    pyplot.ylim(0, parameters.height)
    axis = pyplot.gca()
    axis.set_axis_bgcolor('black')
    axis.axes.get_xaxis().set_visible(False)
    axis.axes.get_yaxis().set_visible(False)
    return fig, writer


def new_frame():
    pyplot.cla()


def annotate_frame(i, e, average_error):
    pyplot.text(1, parameters.height - 1, "Iteration " + str(i + 1), fontsize=12, color='white')
    pyplot.text(1, parameters.height - 2, "Example " + str(e + 1), fontsize=12, color='white')
    if average_error:
        pyplot.text(7, parameters.height - 1, "Average Error " + str(average_error) + "%", fontsize=12, color='white')
        error_bar(average_error)


def error_bar(average_error):
    border = pyplot.Rectangle((7, parameters.height - 3), 10, 1, color='white', fill=False)
    pyplot.gca().add_patch(border)
    rectangle = pyplot.Rectangle((7, parameters.height - 3), 10 * average_error / 100, 1, color='red')
    pyplot.gca().add_patch(rectangle)
