from skimage import io, morphology
import numpy as np
import cv2
import VesselExtraction as ve


########################################## STEP 1: Prepare the vessels #################################################

# this function detects the junctions in the vessel segmentation
# this includes any crossover points and branch points
# takes as input a numpy array with o's corresponding to non vessel pixels and 1's corresponding to vessel pixels
# outputs a numpy array with o's corresponding to non end point pixels and 1's corresponding to end point pixels
def detect_junctions(image):
    # smooth image
    smoothed_image = morphology.binary_closing(image, morphology.disk(10))
    # then skeletonise
    skeleton = morphology.skeletonize(smoothed_image.astype('bool'))
    img = skeleton.astype('uint8')
    # filters that match the structure of a branch or crossover point
    j1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]])
    j2 = np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]])
    j3 = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]])
    j4 = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0]])
    junction_types = [j1, j2, j3, j4]
    edges = np.zeros(img.shape).astype('uint8')
    # finds all pixels in the skeleton that match the filter
    for j in junction_types:
        for i in range(4):
            structuring_element = np.rot90(j, i)
            hit_or_miss = cv2.morphologyEx(img, cv2.MORPH_HITMISS, structuring_element)
            edges = cv2.bitwise_or(edges, hit_or_miss)
    return edges


# this function removes an area around the vessel junction
# the purpose of this is to easily separate the image into different vessel pieces
# takes as input a numpy array with o's corresponding to non vessel pixels and 1's corresponding to vessel pixels
# takes as input numpy array with o's corresponding to non end point pixels and 1's corresponding to end point pixels
# this is found using detect_junctions()
# take as input area size, an integer describing radius of the circular region surrounding the endpoint to remove
# outputs the image with the junctions separated
def separate_vessel_junctions(image, vessel_junctions, area):
    # get the circular area to remove from the junction
    structure_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (area, area))
    region_to_remove = cv2.filter2D(vessel_junctions, -1, structure_element)
    region_to_remove[region_to_remove > 1] = 1
    img = image.copy()
    # remove the relevant point
    img[region_to_remove == 1] = 0
    # this recursion means any extra junctions caused by the erosion will not be an issue
    while np.any(detect_junctions(img) == 1):
        vessel_junctions = detect_junctions(img)
        img = separate_vessel_junctions(img, vessel_junctions, area)
    return img


# creates a list of numpy array as the input image only containing only one vessel
# takes as input a numpy array with o's corresponding to non vessel pixels and 1's corresponding to vessel pixels
# outputs a list of the different vessel images
def list_vessels(image):
    vessels = []
    # now implement the blob algorithm to label the different vessels
    num_segments, labelled_image, __, centroids = cv2.connectedComponentsWithStats(image)
    for i in range(1, len(centroids) + 1):
        vessel = labelled_image.copy()
        vessel[labelled_image != i] = 0
        vessel[labelled_image == i] = 1
        # make sure the vessel segment is large enough to bother including
        if len(np.where(vessel == 1)[0]) > 100:
            vessels.append(vessel)
    return vessels


# returns a list of the vessel centre-line points as coordinates
# takes as input a vessel as given by one element of list_vessels()
# outputs a list of the vessels pixel coordinates
def ordered_vessel_coordinates(vessel):
    smoothed_image = morphology.binary_closing(vessel, morphology.disk(10))
    skeleton = morphology.skeletonize(smoothed_image.astype('bool')).astype('uint8')
    end_points = np.where(ve.detect_end_points(skeleton) == 1)
    coordinates = []
    # there shouldn't be more than two end points
    # but if there is we cannot proceed with this method
    if len(end_points[0]) == 2:
        # determine a start and end point
        start_point = np.array([end_points[0][0], end_points[1][0]])
        end_point = np.array([end_points[0][1], end_points[1][1]])
        current_point = start_point
        while np.any(current_point != end_point):
            # get coordinates of current point
            coordinates.append(current_point)
            # find neighbour
            # remove current_point from the image
            skeleton[current_point[0], current_point[1]] = 0
            # update current point
            end_points = np.where(ve.detect_end_points(skeleton) == 1)
            if len(end_points[0]) == 2:
                p1 = np.array([end_points[0][0], end_points[1][0]])
                p2 = np.array([end_points[0][1], end_points[1][1]])
                current_point = p1
                if np.all(p1 == end_point):
                    current_point = p2
            else:
                # if we reach the end point we are done
                current_point = end_point
                coordinates.append(current_point)
    return coordinates


# this function take a list of coordinates and translates and rotates then so the start point is at the origin
# and the line passing through the start and end point is parallel to the x-axis
# takes as input a list of vessel coordinates as given by ordered_vessel_coordinates()
# outputs a list of the vessels pixel coordinates
def rotate_coords(vessel):
    v = np.array(vessel)
    start_x=v[0,0]
    start_y=v[0,1]
    end_x=v[-1,0]
    end_y=v[-1,1]
    # move origin
    x=v[:,0] - start_x
    y=v[:,1] - start_y
    #rotate
    angle = np.arctan2(start_y - end_y, start_x - end_x)
    qx = -np.cos(angle) * x - np.sin(angle) * y
    qy = np.sin(angle) * x - np.cos(angle) * y
    return [np.array([qx[i], qy[i]]) for i in range(v.shape[0])]

########################################################################################################################


########################################### STEP 2: extract vessel diameter ############################################

# input a binary image with just the vessel segment True

# extraction of diameter of a segmented vessel based on the skeleton length and vessel area
# takes as input a numpy array with o's corresponding to non vessel pixels and 1's corresponding to vessel pixels
# outputs an approximation of the vessel diameter
def extract_diameter(vessel):
    # RISA method
    # skeletonise vessel
    skeleton = morphology.skeletonize(vessel.astype('bool'))
    # count pixels in skeleton
    length = len(np.argwhere(skeleton))
    # count pixels in vessel
    area = len(np.argwhere(vessel))
    # width = total pixels in vessel / skeleton length
    width = area/length
    return width

####################################################################################################


################### STEP 3: Extract image tortuosity ####################


def vessel_length(samples):
    length = 0
    for i in range(len(samples)-1):
        length += np.sqrt(np.sum(np.square(samples[i + 1] - samples[i])))
    return length


# calculate the angle based tortuosity as in Poletti et al. (2012)
# takes as input a list of coordinates of the vessel as produced by ordered_vessel_coordinates() and
# rotate_coords(vessel)
# takes as input the distance between samples to take from the vessel
# outputs the vessel tortuosity and vessel length
def angle_based_tortuosity(vessel, dist):
    samples = [vessel[i] for i in range(len(vessel)) if i % dist == 0]
    length = vessel_length(samples)
    vectors = []
    angles = 0
    for i in range(len(samples) - 1):
        vectors.append(samples[i + 1] - samples[i])
    for i in range(len(vectors) - 1):
        angles += np.square(np.arctan2(vectors[i + 1][1], vectors[i + 1][0]) - np.arctan2(vectors[i][1], vectors[i][0]))
    return (1/length) * angles, length


# splits the vessel into curves
# takes as input the vessel coordinates and distance between samples
# outputs a list of tuples corresponding to the coordinates of the maximum and minimum points on the curve paired with
# the index of the sample
def get_turn_curves(vessel, dist):
    # firstly separate the vessel coordinates into twist start and end points
    samples = [vessel[i] for i in range(len(vessel)) if i % dist == 0]
    grads = []
    turn = [(vessel[0], 0)]
    # compute gradient between each of the sample points
    for i in range(len(samples) - 1):
        if (samples[i][0] - samples[i + 1][0]) == 0:
            grad = np.inf
        else:
            grad = (samples[i][1] - samples[i + 1][1])/(samples[i][0] - samples[i + 1][0])
        grads.append(grad)
    # list e
    for i in range(len(grads) - 1):
        # this is just any point of inflection at the moment
        non_zero = grads[i] != 0 and grads[i + 1] != 0
        non_inf = grads[i] != np.inf and grads[i + 1] != np.inf
        if non_zero and non_inf and (grads[i] * grads[i + 1] < 0):
            turn.append((samples[i + 1], i+1))
    turn.append((samples[-1], len(grads)))
    return turn


# calculates the twist based tortuosity as used by Poletti et al. (2012) and proposed by Grisan et al. (2008)
# takes as input the vessel coordinates and distance between samples
# outputs the vessel length and twist based tortuosity
def twist_based_tortuosity(vessel, dist):
    samples = [vessel[i] for i in range(len(vessel)) if i % dist == 0]
    # get the turn curves
    turns = get_turn_curves(vessel, dist)
    # get the vessel length
    length = vessel_length(samples)
    # get the number of turns
    T = len(turns) - 1
    track = 0
    for i in range(len(turns) - 1):
        t1 = turns[i]
        t2 = turns[i + 1]
        sample_index1 = t1[1]
        sample_index2 = t2[1]
        # calculate turn curve arc length
        arc_length = 0
        for j in range(sample_index1, sample_index2):
            arc_length += np.sqrt(np.sum(np.square(samples[j] - samples[j + 1])))
        # calculate turn curve chord length
        chord_length = np.sqrt(np.sum(np.square(turns[i][0] - turns[i + 1][0])))
        # calculate ratio
        ratio = arc_length/chord_length
        # update value
        track += ratio - 1
    return (1/length) * (T - 1)/T * track, length

####################################################################################################


################################### Leaf node count ###############################################

# calculate the leaf node count of the vessels as used by Nisha et al. (2019)
# takes as input a numpy array with o's corresponding to non vessel pixels and 1's corresponding to vessel pixels
# return the leaf node count of the image
def vessel_leaf_nodes(image):
    skeleton = morphology.skeletonize(image.astype('bool')).astype('uint8')
    leaf_node_count = len(np.where(ve.detect_end_points(skeleton) == 1)[0])
    return leaf_node_count

####################################################################################################


###################################### Image level metrics #########################################

# combines the feature extraction methods to provide image level features
# the image level tortuosity metrics proposed were by Poletti et al. (2012)
# takes as input a numpy array with o's corresponding to non vessel pixels and 1's corresponding to vessel pixels
# takes as input the distance between samples and removal area of as required by separate_vessel_junctions() and
# angle_based_tortuosity() and twist_based_tortuosity()
def image_level_features(image, dist, area):
    junctions = detect_junctions(image)
    sep_vessels = np.uint8(separate_vessel_junctions(image, junctions, area))
    vessels = list_vessels(sep_vessels)
    diameters = []
    num_vessels = 0
    it1 = 0
    it2 = 0
    it3 = 0
    it4 = 0
    for i, vessel in enumerate(vessels):
        # extraction of vessel coordinates
        v = ordered_vessel_coordinates(vessel)
        # make sure the vessel is longer than the samples being taken!
        if len(v) > dist:
            # rotation of vessel
            v = rotate_coords(v)
            # vessel level tortuosity calculation
            phi, length = angle_based_tortuosity(v, dist)
            tau, length = twist_based_tortuosity(v, dist)
            it1 += tau
            it2 += tau * length
            it3 += phi
            it4 += phi * length
            # diameter calculation
            diameter = extract_diameter(vessel)
            diameters.append(diameter)
            num_vessels += 1
        else:
            num_vessels -= 1
    # next calculate the diameter based features
    if num_vessels > 0:
        k = 1 / num_vessels
        max_diameter = np.max(diameters)
        min_diameter = np.min(diameters)
        avg_diameter = np.mean(diameters)
        # now finally the leaf node count
        leaf_nodes = vessel_leaf_nodes(image)
    else:
        max_diameter = np.nan
        min_diameter = np.nan
        avg_diameter = np.nan
        # now finally the leaf node count
        leaf_nodes = np.nan

    return max_diameter, min_diameter, avg_diameter, k * it1, k * it2, k * it3, k * it4, leaf_nodes

####################################################################################################

