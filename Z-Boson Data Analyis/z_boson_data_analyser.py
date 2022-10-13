"""
Z Boson Data Analyser
Qi Nohr Chen 11/10/21

Program that will take in file data and validate it to plot a contour as well
as a result plot. Additionally will calculate the mass, lifetime and width
parameters with their uncertainties. Program will save the plots in the same
directory.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import scipy.constants as pc

BOSON_WIDTH_PAIR = 0.08391 #Constant used to calculte
#The files, which can be changed by the user are found here
#Change the delimiter accordingly
#Change the skip header accordingly
FILE_1 = "z_boson_data_1.csv"
FILE_2 = "z_boson_data_2.csv"
DELIMITER = ","
SKIP_HEADER = 1
#User may also manually change the guesses
GUESSES = (90, 3)
#Empty sets that are used as place holder, please don not change them
MASS_EMPTY = [0]
WIDTH_EMPTY = [0]
CHI_SQUARED_EMPTY = [0]
COUNTER = 0


def open_file(file_name):
    """
    This function will open the file using np.genfrom.txt. The delimiter can
    be cahnged at the top as well as the amount of headers the function skips.
    Args:
        file_name: file
    returns:
        opened file
    """
    return np.genfromtxt(file_name, delimiter=DELIMITER,
                         skip_header=SKIP_HEADER)

def function(x_variables, mass_parameter, boson_width_parameter):
    """
    This function is the theoretical equation used to calculate the
    cross section of the z boson. It can take in observed x_variables,
    which in this case are the energies from the file, or values created by
    linspace.
    Args:
        x_variable: data_array
        mass_parameter: float
        boson_width_parameter: float
    Returns:
        Cross section equation and calculation for each x_variable
    """
    return ((12*np.pi/mass_parameter**2)
            *(x_variables**2/((x_variables**2-mass_parameter**2)**2
                              +(mass_parameter**2)*boson_width_parameter**2))
            *BOSON_WIDTH_PAIR**2)*389400

def combine_data(data1, data2):
    """
    As two files have been given in, they need to be combined. This function
    will do exactly that.
    Args:
        data1: array
        data2: array
    Returns:
        combining_data: array
    """
    combining_data = np.vstack((data1, data2))
    return combining_data

def validate_file(opened_file):
    """
    Function will validate a file that has been inputed and remove them from
    from the data sat using a for loop and a try and except.
    Args:
        opened_file: array
    Raises:
        ValueError: If something  can't be converted into a float
        TypeError: If something is an incorrect type
    Returns:
        data: array
    """
    data = np.zeros((0, 3))
    for line in opened_file:
        try:
            temporary = np.array([float(line[0]),
                                  float(line[1]), float(line[2])])
            data = np.vstack((data, temporary))
        except ValueError:
            print("Invalid Data found:", line)
        except TypeError:
            print(("Invalid Data found:", line))
    return data

def nan_removal(validated_data):
    """
    This function will take in the validated data and remove all the nan
    values in it. This will then which will in turn return a fully valid
    data set that can be used for plotting.
    Args:
        validated_data: array
    Returns:
        non_nan_array: array
    """
    non_nan_array = np.zeros((0, 3))
    for line in validated_data:
        if any(np.isnan(line)) is True:
            print("Invalid data:", line)
        else:
            non_nan_array = np.vstack((non_nan_array, line))
    return non_nan_array

def remove_large_outlier(nan_removed_data):
    """
    The function will take care of any extremely large outliers that are
    present in the data set using a logical statement and the standard
    deviation of the data set. It takes in the fully validated file with
    nothing but floats.
    Args:
       nan_removed_data: array
    Returns:
        array_outlier_removal: array
    """
    array_outlier_removal = np.zeros((0, 3))
    for line in nan_removed_data:
        if line[1] < np.std(nan_removed_data[:, 1]):
            array_outlier_removal = np.vstack((array_outlier_removal, line))
        else:
            print("outlier removed", line)
    return array_outlier_removal

def zero_filter(unfiltered_data):
    """
    Function will filter out all the 0 values in the uncertainties.
    Args:
        unflitered_data: array
    Returns:
        filtered_zero: array
    """
    filtered_zero = unfiltered_data[unfiltered_data[:, 2] != 0]
    return filtered_zero

def data_sorting(unsorted_data):
    """
    The fully validated data will be unsorted. This function will ensure
    that the data inputted will be sorted according to the lowest to highest
    energy values.
    Args:
        unsorted_data: array
    Returns:
        sort_data: array
    """
    sorted_indexes = np.argsort(unsorted_data[:, 0])
    sort_data = unsorted_data[sorted_indexes, :]
    return sort_data

def chi_squared(parameter_1_and_2):
    """
    Function will use two guess parameters, the observed energy values
    (known as x_values in this function), the observed cross sectional
    values (y_values) and the observed uncertainties.
    Args:
        parameter_1_and_2: tuple
    Returns:
        chi_square_function: float
    """
    parameter_guess_1 = parameter_1_and_2[0]
    parameter_guess_2 = parameter_1_and_2[1]
    chi_square_function = np.sum(((function(x_values, parameter_guess_1,
                                            parameter_guess_2) - y_values)/
                                  uncertainties)**2)
    return chi_square_function

def remove_outliers(data_array, mass_coefficient, boson_width_coefficient):
    """
    Function removed outliers after the mass coefficeint and boson width
    coefficients had been found.
    Args:
        data_array: array
        mass_coefficient: float
        boson_width_coefficient: float
    returns:
        new_data: array
    """
    new_data = np.zeros((0, 3))
    for line in data_array:
        if np.sqrt(((line[1]- function(line[0],
                                       mass_coefficient,
                                       boson_width_coefficient))
                    )**2) < 3*line[2]:
            new_data = np.vstack((new_data, line))
    return new_data

def mesh_array(coefficient_1, coefficient_2):
    """
    Function will create two mesh array for the mass and the width, using
    the found parameters and creating a range in between them. Coefficient_1
    is the mass coefficient where as coefficient_2 is the width coefficient.
    Args:
        coefficient_1: float
        coefficient_2: float
    Returns:
        mass_xx: mesh array
        width_yy: mesh array
    """
    #Sensitivity can be changed in the linspace (Standard sensitivity:200)
    mass_trials = np.linspace(coefficient_1-0.05, coefficient_1+0.05, 200)
    width_trials = np.linspace(coefficient_2-0.02, coefficient_2+0.02, 200)
    mass_xx, width_yy = np.meshgrid(mass_trials, width_trials)
    return mass_xx, width_yy

def empty_z_mesh(mesh_of_mass, mesh_of_width):
    """
    This function uses the dimensions of the mass mesh array and the width
    mesh array to create an empty array for the chi squared mesh, which
    will be along the z-axis.
    Args:
        mesh_of_mass: mesh array
        mesh_of_width: mesh array
    Returns:
        emmpty_mesh: mesh array
    """
    empty_mesh = np.zeros((len(mesh_of_mass[0, :]), len(mesh_of_width[:, 0])))
    return empty_mesh

def chi_square_mesh(meshed_mass, meshed_width, mesh_empty):
    """
    Function will calculate the chi square at each point in the mesh and
    insert it into the empty chi square mesh to create a new mesh array
    that is ready to be plotted. The function has redefined mesh_empty
    with values, such that the return is NOT an empty array, but an array
    with values.
    Args:
        meshed_mass: mesh array
        meshed_width: mesh array
        mesh_empty: mesh array
    Returns:
        mesh_empty: mesh array
    """
    for i in range(len(mesh_empty[0])):
        for j in range(len(mesh_empty[0])):
            mesh_empty[i, j] = chi_squared((meshed_mass[i, j],
                                            meshed_width[i, j]))
    return mesh_empty #Redefine the inputted array

def file_converter(first_file, second_file):
    """
    Function uses all previous file confirmation/conversion/validation
    functions such that they are ready for plotting.
    Args:
        first_file: file
        second_file: file
    Returns:
        x_values_converted: array
        y_values_converted: array
        error_converted: array
        new_data: array
    """
    raw_data_1 = open_file(first_file)
    raw_data_2 = open_file(second_file)
    combined_raw_data = validate_file(combine_data(raw_data_1, raw_data_2))
    combined_data = (remove_large_outlier((nan_removal(combined_raw_data))))
    new_data = zero_filter((data_sorting(combined_data)))
    x_values_converted = new_data[:, 0]
    y_values_converted = new_data[:, 1]
    error_converted = new_data[:, 2]
    return x_values_converted, y_values_converted, error_converted, new_data

def contour_plotting(x_mesh, y_mesh, z_mesh):
    """
    Function plots contour of the chi squared and will also find the
    uncertainties of the coefficients of mass and width. x_mesh is the mesh
    of the mass, y_mesh is the mesh of the width and z_mesh is the mesh of the
    chi squared.
    Args:
        x_mesh: mesh array
        y_mesh: mesh array
        z_mesh: mesh array
    Returns:
        uncertainty_mass: float
        uncertainty_width: float
        Plots: contour plot
        Saves contour plot as: 'contour_plot_mass_vs_width_vs_chi_squared.png'
    """
    fig_2 = plt.figure()
    axes_2 = fig_2.add_subplot(111)
    contour_plot = axes_2.contour(mesh_mass,
                                  mesh_width, chi_square_mesh(x_mesh,
                                                              y_mesh,
                                                              z_mesh),
                                  [CHI_SQUARE+1])
    axes_2.clabel(contour_plot)
    axes_2.set_title('Chi_Squared_contour')
    axes_2.set_xlabel('Mass (GeV/c^2)')
    axes_2.set_ylabel('Width (nb)')
    point_1 = contour_plot.collections[0].get_paths()[0]
    coordinate_point_1 = point_1.vertices
    mass_max = np.max(coordinate_point_1[:, 0])
    mass_min = np.min(coordinate_point_1[:, 0])
    uncertainty_mass = (mass_max-mass_min)/2
    width_max = np.max(coordinate_point_1[:, 1])
    width_min = np.min(coordinate_point_1[:, 1])
    uncertainty_width = (width_max-width_min)/2
    plt.savefig('contour_plot_mass_vs_width_vs_chi_squared.png',
                bbox_inches="tight",
                dpi=300)
    plt.show()
    return uncertainty_mass, uncertainty_width

def reduced_chi_squared(chi_square_value, data_set):
    """
    Function calculates the reduced chi squared and uses a data set to
    calculate the degrees of freedom.
    Args:
        chi_square_value: float
        data_set: array
    Returns:
        reduce_chi_squared: float
    """
    number_of_data_points = len(data_set)
    degrees_of_freedom = number_of_data_points - 2
    reduced_chi_square = chi_square_value/degrees_of_freedom
    return reduced_chi_square

def life_time(width_value, uncertainty_width):
    """
    Calculates the life time of the z boson as well as the uncertainty using
    the width calculated and the uncertainty of the calculated width.
    Args:
        width_value: float
        uncertainty_width: float
    Returns:
        life_time_calculation: float
        final_uncertainty: float
    """
    width_value = width_value * pc.e * 10**9
    uncertainty_width = uncertainty_width * pc.e * 10**9
    life_time_calculation = pc.hbar/width_value
    fractional_uncertainty = uncertainty_width/width_value
    inverse_width_uncertainty = (1/width_value) * fractional_uncertainty
    final_uncertainty = pc.hbar * inverse_width_uncertainty
    return life_time_calculation, final_uncertainty

def combined_plot(combined_data, mass_final, mass_uncertainty, width_final,
                  width_uncertainty, life_time_value,
                  life_time_uncertainty_value,
                  reduced_chi_squared_calculated):
    """
    Function that will plot the values found as well as the correct plot that
    has been validated and has it's outliers removed from the data set. Will
    make use of the 'function' to create a line of best fit to fit onto the
    data.
    Args:
        combined_data: array
        mass_final: float
        mass_uncertainty: float
        width_final:float
        width_uncertainty: float
        life_time_value: float
        life_time_uncertainty: float
        reduced_chi_squared_calculated: float
    Returns:
        Plots: Energy against Cross Section
        Saved Plot under the name 'Z_Boson_Energy_vs_Cross_Section_plot.png'
    """
    mass_string = "Mass = {0:.4g} +/- {1:4.3f} Gev/c^2".format(
        mass_final, mass_uncertainty)
    width_string = "Width = {0:.4g} +/- {1:4.3f} GeV".format(
        width_final, width_uncertainty)
    life_time_string = "Life Time = {0:3.2e} +/- {1:3.2e} s".format(
        life_time_value, life_time_uncertainty_value)
    reduced_chi_square_string = "Reduced Chi Square = {0:.3g}".format(
        reduced_chi_squared_calculated)
    fig = plt.figure()
    axes_plot = fig.add_subplot(111)
    axes_plot.set_title("Z Boson Energy vs Cross Section", fontsize=14)
    axes_plot.set_xlabel('Energy in GeV', fontsize=14)
    axes_plot.set_ylabel('Cross Section in nb', fontsize=14)
    axes_plot.plot(x_values, function(x_values, 91, 2.5),
                   label="Best Fit Line")
    axes_plot.errorbar(combined_data[:, 0], combined_data[:, 1],
                       yerr=combined_data[:, 2], fmt='o', alpha=0.3,
                       color="r", label="Observed Data")
    plt.figtext(0.5, -0.1, mass_string, fontsize=14)
    plt.figtext(0.5, -0.15, width_string, fontsize=14)
    plt.figtext(0.5, -0.2, life_time_string, fontsize=14)
    plt.figtext(0.5, -0.25, reduced_chi_square_string, fontsize=14)
    plt.savefig('Z_Boson_Energy_vs_Cross_Section_plot.png',
                bbox_inches="tight",
                dpi=300)
    plt.legend()
    plt.show()

#MAIN CODE: This will run the plots and do all the calculations.
#Nothing needs to be changed here
start_time = time.time()
x_values, y_values, uncertainties, sorted_data = file_converter(FILE_1,
                                                                FILE_2)
x_values = sorted_data[:, 0]
y_values = sorted_data[:, 1]
uncertainties = sorted_data[:, 2]
#Minimistion routine
result = fmin(chi_squared, GUESSES, full_output=True, disp=0)
final_data = remove_outliers(sorted_data, result[0][0], result[0][1])
MASS_EMPTY.append(result[0][0])
WIDTH_EMPTY.append(result[0][1])
x_values = final_data[:, 0]
y_values = final_data[:, 1]
uncertainties = final_data[:, 2]
CHI_SQUARED_EMPTY.append(result[1])
COUNTER += 1
while MASS_EMPTY[-1] != MASS_EMPTY[-2] and WIDTH_EMPTY[-1] != WIDTH_EMPTY[-2]:
    result = fmin(chi_squared, GUESSES, full_output=True, disp=0)
    final_data = remove_outliers(sorted_data, MASS_EMPTY[-1], WIDTH_EMPTY[-1])
    MASS_EMPTY.append(result[0][0])
    WIDTH_EMPTY.append(result[0][1])
    CHI_SQUARED_EMPTY.append(result[1])
    COUNTER += 1
    x_values = final_data[:, 0]
    y_values = final_data[:, 1]
    uncertainties = final_data[:, 2]

print("This took", COUNTER, "iterations to complete and check minimisation")
#Calculated values
MASS = MASS_EMPTY[-1]
WIDTH = WIDTH_EMPTY[-1]
CHI_SQUARE = CHI_SQUARED_EMPTY[-1]
REDUCED_CHI_SQUARED = reduced_chi_squared(CHI_SQUARE, x_values)
#Plotting of contour
mesh_mass, mesh_width = mesh_array(MASS, WIDTH)
mesh_chi_square = chi_square_mesh(mesh_mass, mesh_width,
                                  empty_z_mesh(mesh_mass, mesh_width))
#Calculated uncertainties and life time
ERROR_MASS, ERROR_WIDTH = contour_plotting(mesh_mass, mesh_width,
                                           empty_z_mesh(mesh_mass,
                                                        mesh_width))
LIFE_TIME, ERROR_LIFE_TIME = life_time(WIDTH, ERROR_WIDTH)
#Plotting of results
combined_plot(final_data, MASS, ERROR_MASS, WIDTH, ERROR_WIDTH, LIFE_TIME,
              ERROR_LIFE_TIME, REDUCED_CHI_SQUARED)
execution_time = (time.time() - start_time)
#Execution time calculated
print("This code took", execution_time, "To run")
