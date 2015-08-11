import parameters
from matplotlib import pyplot, animation, rcParams


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
    rcParams['font.size'] = 12
    rcParams['text.color'] = 'white'
    return fig, writer


def new_frame():
    pyplot.cla()


def annotate_frame(i, e, average_error, example):
    pyplot.text(1, parameters.height - 1, "Iteration #" + str(i + 1))
    pyplot.text(1, parameters.height - 2, "Training example #" + str(e + 1))
    pyplot.text(1, parameters.output_y_position, "Desired output:")
    pyplot.text(1, parameters.output_y_position - 1, str(example.output))
    pyplot.text(1, parameters.bottom_margin + 1, "Inputs:")
    pyplot.text(1, parameters.bottom_margin, str(example.inputs))
    if average_error:
        error_bar(average_error)


def error_bar(average_error):
    pyplot.text(parameters.error_bar_x_position, parameters.height - 1, "Average Error " + str(average_error) + "%")
    border = pyplot.Rectangle((parameters.error_bar_x_position, parameters.height - 3), 10, 1, color='white', fill=False)
    pyplot.gca().add_patch(border)
    rectangle = pyplot.Rectangle((parameters.error_bar_x_position, parameters.height - 3), 10 * average_error / 100, 1, color='red')
    pyplot.gca().add_patch(rectangle)
