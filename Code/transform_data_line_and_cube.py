#=================================================================================
#=================================================================================
# Script:"transform_data_line_and_cube"
# Date: 2022-09-21
# Implemented by: Johannes Borgqvist
# Description:
# The script generates data from a linear model y(t)=Ct
# and a cubic model y=C*t^3. It then transforms the data
# by the rotation symmetry of the linear model and the
# scaling symmetry of the cubic model. Then, we compare
# how the fit to transformed data agrees with the theoretical
# prediction.
#=================================================================================
#=================================================================================
#=================================================================================
#=================================================================================
# Import Libraries
#=================================================================================
#=================================================================================
from numpy import * # For numerical calculations,
from scipy.optimize import curve_fit # For fitting curves to data
import matplotlib.pyplot as plt # For plotting,
import pandas as pd # For saving data

#=================================================================================
#=================================================================================
# Functions
#=================================================================================
#=================================================================================
# Function 1: "plot_LaTeX_2D"
# The function takes the following input:
# 1. "t" being the list of t-values or input values,
# 2. "y" being the list of y-values or output values,
# 3. "file_str" being the string defining where the output
# file with format tex is stored,
# 4. "plot_str" is the string defining properties such as
# colour and linewidth,
# 5. "legend_str" is a string containing the legend of the plot.
# The function then creates a tex file which can be plotted
# in LaTeX using pgfplots. 
def plot_LaTeX_2D(t,y,file_str,plot_str,legend_str):
    # Open a file with the append option
    # so that we can write to the same
    # file multiple times
    f = open(file_str, "a")
    # Create a temporary string which
    # is the one that does the plotting.
    # Here we incorporate the input plot_str
    # which contains the color, and the markers
    # of the plot at hand
    if len(legend_str)==0:
        temp_str = "\\addplot[\nforget plot,\n" + plot_str+ "\n]\n"
    else:
        temp_str = "\\addplot[\n" + plot_str+ "\n]\n"
    # Add the coordinates
    temp_str += "coordinates {%\n"
    # Loop over the input files and add
    # them to the file
    for i in range(len(t)):
        temp_str += "(" + str(t[i]) + "," + str(y[i]) + ")\n"
    # The plotting is done, let's close the shop    
    temp_str += "};\n"
    # Add a legend if one is provided
    if len(legend_str) > 0:
        temp_str += "\\addlegendentry{" + legend_str + "}\n"
    # Finally, we write the huge string
    # we have created
    f.write("%s"%(temp_str))
    # Close the file
    f.close()
# Function 2: linear_model
def linear_model(x, k):
    return k*x
# Function 3: cubic_model
def cubic_model(x, k):
    return k*(x**3)
#=================================================================================
#=================================================================================
# Generate data
#=================================================================================
#=================================================================================
# Define the number of data points
num_points = 5
# Define the noise level
sigma_lin = 0.1
sigma_cube = 2.5
# Define the errors of the linear model
errors_linear = random.normal(0, sigma_lin, size=(num_points,))
# Define the errors of the cubic model
errors_cubic = random.normal(0, sigma_cube, size=(num_points,))
# Define our constant C
C_lin = 0.1
C_cube = 0.1
# We do not start at zero I suppose so set a base line
base_line_linear = 2
base_line_cubic = 2
# Generate data from the linear model
data_lin = array([C_lin*(index+base_line_linear)+error for index,error in enumerate(errors_linear)])
t_lin = array([index+base_line_linear for index,error in enumerate(errors_linear)])
# Save the linear data to file
file_name_lin = "../Data/linear_data_sigma_" + str(round(sigma_lin,3)).replace(".","p") + ".csv"
pd.DataFrame(data = array([list(t_lin), list(data_lin)]).T,index=["Row "+str(temp_index+1) for temp_index in range(5)],columns=["t", "y(t)"]).to_csv(file_name_lin)
# Generate data from the cubic model
data_cube = array([C_cube*(index+base_line_cubic)**3+error for index,error in enumerate(errors_cubic)])
t_cube = array([index+base_line_cubic for index,error in enumerate(errors_cubic)])
# Save the cubic data to file
file_name_cube = "../Data/cubic_data_sigma_" + str(round(sigma_cube,3)).replace(".","p") + ".csv"
pd.DataFrame(data = array([list(t_cube), list(data_cube)]).T,index=["Row "+str(temp_index+1) for temp_index in range(5)],columns=["t", "y(t)"]).to_csv(file_name_cube)
#=================================================================================
#=================================================================================
# Fit models to data
#=================================================================================
#=================================================================================
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# LINEAR DATA
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# Fit the linear model to linear data
popt_data_lin_fit_lin, pcov_data_lin_fit_lin = curve_fit(linear_model, t_lin, data_lin)
# Calculate the sum of squares, SS
res_data_lin_fit_lin = linear_model(t_lin,*pcov_data_lin_fit_lin)-data_lin
SS_data_lin_fit_lin = 0
for res in res_data_lin_fit_lin:
    SS_data_lin_fit_lin += res**2
SS_data_lin_fit_lin = SS_data_lin_fit_lin/(len(t_lin)-1)
# Extract parameter
C_lin_opt_data_lin = popt_data_lin_fit_lin[0]
# Generate the optimal line
lin_opt_data_lin = array([C_lin_opt_data_lin*(index+base_line_linear) for index,error in enumerate(errors_linear)])
# Fit the cubic model to linear data
popt_data_lin_fit_cube, pcov_data_lin_fit_cube = curve_fit(cubic_model, t_lin, data_lin)
# Calculate the sum of squares, SS
res_data_lin_fit_cube = cubic_model(t_lin,*pcov_data_lin_fit_cube)-data_lin
SS_data_lin_fit_cube = 0
for res in res_data_lin_fit_cube:
    SS_data_lin_fit_cube += res**2
SS_data_lin_fit_cube = SS_data_lin_fit_cube/(len(t_lin)-1)
# Extract parameter
C_cube_opt_data_lin = popt_data_lin_fit_cube[0]
# Generate the optimal line
t_cube_opt_data_lin = linspace(base_line_linear,base_line_linear+len(errors_linear)-1,20)
cube_opt_data_lin = array([C_cube_opt_data_lin*(t_temp)**3 for t_temp in t_cube_opt_data_lin])
#=================================================================================
#=================================================================================
# Plot the symmetries
#=================================================================================
#=================================================================================
# Set all parameters to tex
plt.rcParams['text.usetex'] = True
# Define a figure window with two subfigures
fig_1, axs_1 = plt.subplots(1, 2, constrained_layout=True, figsize=(20, 8))
# The linear model
# The original solution
axs_1[0].plot(t_lin,data_lin, 'o', label="Data linear, $e_i\\sim\\mathcal{N}(0,\\sigma),\quad\\sigma="+ str(round(sigma_lin,3))+ "$" ,color=(0/256,0/256,0/256),linewidth=3.0)
axs_1[0].plot(t_lin,lin_opt_data_lin, '-', label="Linear opt, $SS= " + str(round(SS_data_lin_fit_lin,7)) +"$" ,color=(0/256,68/256,27/256),linewidth=3.0)
axs_1[0].plot(t_cube_opt_data_lin,cube_opt_data_lin, '-', label="Cube opt, $SS= " + str(round(SS_data_lin_fit_cube,7)) +"$" ,color=(103/256,0/256,31/256),linewidth=3.0)
axs_1[0].grid()
# Set the limits
#axs_1[0].set_xlim([-1, 1])
#axs_1[0].set_ylim([-1, 1])
# Legends and axes labels
axs_1[0].legend(loc='best',prop={"size":20})
axs_1[0].set_ylabel('Explanatory variable, $y(t)$',fontsize=25)
axs_1[0].set_xlabel('Response variable, $t$',fontsize=25)
# Change the size of the ticks
axs_1[0].tick_params(axis='both', which='major', labelsize=20)
axs_1[0].tick_params(axis='both', which='minor', labelsize=20)
# Title and saving the figure
axs_1[0].set_title("Data generated by the linear model, $y(t)="+ str(C_lin) + "\\;t$",fontsize=30,weight='bold');
# The cubic model
# The original cubic curve
axs_1[1].plot(t_cube,data_cube, '<', label="Data cube" ,color=(0/256,0/256,0/256),linewidth=3.0)
axs_1[1].grid()
# Set the limits
#axs_1[1].set_xlim([-2, 2])
#axs_1[1].set_ylim([-2, 2])
# Legends and axes labels
axs_1[1].legend(loc='best',prop={"size":20})
axs_1[1].set_ylabel('Explanatory variable, $y(t)$',fontsize=25)
axs_1[1].set_xlabel('Response variable, $t$',fontsize=25)
# Change the size of the ticks
axs_1[1].tick_params(axis='both', which='major', labelsize=20)
axs_1[1].tick_params(axis='both', which='minor', labelsize=20)
# Title and saving the figure
axs_1[1].set_title("Data generated by the cubic model, $y(t)="+ str(C_cube) + "\\;t^3$",fontsize=30,weight='bold');
# Save the figure
fig_1.savefig('../Figures/data_line_and_cube.png')
# Show the figure
plt.show()
