#import some stuff

def slice_n_dice(image_array, cube_size):
    """
    Takes a preprocessed image and sliced into a bunch of cubes of various sizes and locations for easier processing
    :param image_array: numpy array of the diacom image
    :param cube_size: how big of a cube do you want
    :return:
    """

def inspect_cube(cube_array):
    """
    takes a cube array subset of a 3D image and looks for something tumor-y
    :param cube_array:
    :return:
    """
    # is it tube shaped or not:
    # find the center of mass
    # find the mean distance for equally (with some buffer) dense pixels
    # is the mean equal in all/most directions?
    # are there bits touching the edges of the cube?
    # How big are the cross sections that intersect the cube wall

