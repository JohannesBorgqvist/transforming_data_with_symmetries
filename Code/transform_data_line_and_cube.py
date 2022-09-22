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
import pandas as pd # For saving and reading data

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
# Function 4: read_generated_data
def read_generated_data(file_str):
    # Read the data into a pandas dataframe
    df = pd.read_csv(file_str)
    # Convert the data frame into an array
    arr = df.to_numpy()
    # Remove the column names "t" and "y(t)" from this array
    new_array = delete(arr.T, 0, 0).T
    # Extract the time and the function value
    t = new_array[:,0]
    y = new_array[:,1]
    # Return these two
    return t, y
#=================================================================================
#=================================================================================
# Overall properties of the linear model and the cube
#=================================================================================
#=================================================================================
# Define our constant C
C_lin = 0.1
C_cube = 0.01
# Define the noise level
sigma_lin = 0.1
sigma_cube = 0.1
#=================================================================================
#=================================================================================
# Generate data or read data from a file.
# Uncomment this if you want to generate data
#=================================================================================
#=================================================================================
# Define the number of data points
#num_points = 5
# Define the errors of the linear model
#errors_linear = random.normal(0, sigma_lin, size=(num_points,))
# Define the errors of the cubic model
#errors_cubic = random.normal(0, sigma_cube, size=(num_points,))
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# Generate data from the linear model
#t_lin = linspace(2,6,num_points,endpoint=True)
#data_lin = array([C_lin*t_temp+errors_linear[index] for index,t_temp in enumerate(t_lin)])
# Define the file name
#file_name_lin = "../Data/new_data_linear_data_sigma_" + str(round(sigma_lin,3)).replace(".","p") + ".csv"
# Save the linear data to file
#pd.DataFrame(data = array([list(t_lin), list(data_lin)]).T,index=["Row "+str(temp_index+1) for temp_index in range(5)],columns=["t", "y(t)"]).to_csv(file_name_lin)
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# Generate data from the cubic model
#t_cube = linspace(3.5,4.5,num_points,endpoint=True)
#data_cube = array([C_cube*(t_temp)**3+errors_cubic[index] for index,t_temp in enumerate(t_cube)])
# Define the file name
#file_name_cube = "../Data/new_data_cubic_data_sigma_" + str(round(sigma_cube,3)).replace(".","p") + ".csv"
# Save the cubic data to file
#pd.DataFrame(data = array([list(t_cube), list(data_cube)]).T,index=["Row "+str(temp_index+1) for temp_index in range(5)],columns=["t", "y(t)"]).to_csv(file_name_cube)
#=================================================================================
#=================================================================================
# Alternatively, we read data from a file
#=================================================================================
#=================================================================================
# Define the file we want to read the data from
cubic_data_str = "../Data/cubic_data_sigma_" + str(round(sigma_cube,3)).replace(".","p") + ".csv"
# Read the data from the file
t_cube, data_cube = read_generated_data(cubic_data_str)
# Define the file we want to read the data from
linear_data_str = "../Data/linear_data_sigma_" + str(sigma_lin).replace(".","p") + ".csv"
# Read the data from the file
t_lin, data_lin = read_generated_data(linear_data_str)
# Calculate the number of points
num_points = len(t_lin)
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
# 1. Fit the linear model to linear data
popt_data_lin_fit_lin, pcov_data_lin_fit_lin = curve_fit(linear_model, t_lin, data_lin)
# Calculate residuals
res_data_lin_fit_lin = linear_model(t_lin,*popt_data_lin_fit_lin)-data_lin
# Calculate the sum of squares, SS
SS_data_lin_fit_lin = sum(array([res**2 for res in res_data_lin_fit_lin]))/(len(t_lin)-1)
# Extract parameter
C_lin_opt_data_lin = popt_data_lin_fit_lin[0]
# Generate the optimal line
lin_opt_data_lin = array([C_lin_opt_data_lin*t_temp for t_temp in t_lin])
#---------------------------------------------------------------------------------
# 2. Fit the cubic model to linear data
popt_data_lin_fit_cube, pcov_data_lin_fit_cube = curve_fit(cubic_model, t_lin, data_lin)
# Calculate residuals
res_data_lin_fit_cube = cubic_model(t_lin,*popt_data_lin_fit_cube)-data_lin
# Calculate the sum of squares, SS
SS_data_lin_fit_cube = sum(array([res**2 for res in res_data_lin_fit_cube]))/(len(t_lin)-1)
# Extract parameter
C_cube_opt_data_lin = popt_data_lin_fit_cube[0]
# Generate the optimal cube
t_cube_opt_data_lin = linspace(t_lin[0],t_lin[-1],20)
cube_opt_data_lin = array([C_cube_opt_data_lin*(t_temp)**3 for t_temp in t_cube_opt_data_lin])
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# CUBIC DATA
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# 1. Fit the linear model to cubic data
popt_data_cube_fit_lin, pcov_data_cube_fit_lin = curve_fit(linear_model, t_cube, data_cube,20)
# Calculate residuals 
res_data_cube_fit_lin = linear_model(t_cube,*popt_data_cube_fit_lin)-data_cube
# Calculate the sum of squares, SS
SS_data_cube_fit_lin = sum(array([res**2 for res in res_data_cube_fit_lin]))/(len(t_cube)-1)
# Extract parameter
C_lin_opt_data_cube = popt_data_cube_fit_lin[0]
# Generate the optimal line
lin_opt_data_cube = array([C_lin_opt_data_cube*t_temp for t_temp in t_cube])
#---------------------------------------------------------------------------------
# 2. Fit the cubic model to cubic data
popt_data_cube_fit_cube, pcov_data_cube_fit_cube = curve_fit(cubic_model, t_cube, data_cube)
# Calculate residuals
res_data_cube_fit_cube = cubic_model(t_cube,*popt_data_cube_fit_cube)-data_cube
# Calculate the sum of squares, SS
SS_data_cube_fit_cube = sum(array([res**2 for res in res_data_cube_fit_cube]))/(len(t_cube)-1)
# Extract parameter
C_cube_opt_data_cube = popt_data_cube_fit_cube[0]
# Generate the optimal cube
t_cube_opt_data_cube = linspace(t_cube[0],t_cube[-1],20)
cube_opt_data_cube = array([C_cube_opt_data_cube*(t_temp)**3 for t_temp in t_cube_opt_data_cube])
#=================================================================================
#=================================================================================
# Plot the symmetries
#=================================================================================
#=================================================================================
# Set all parameters to tex
plt.rcParams['text.usetex'] = True
# Define a figure window with two subfigures
fig_1, axs_1 = plt.subplots(1, 2, constrained_layout=True, figsize=(20, 8))
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# The linear data
axs_1[0].plot(t_lin,data_lin, 'o', label="Data linear, $e_i\\sim\\mathcal{N}(0,\\sigma),\quad\\sigma="+ str(round(sigma_lin,3))+ "$" ,color=(0/256,0/256,0/256),linewidth=3.0)
# The linear model fitted to linear data
axs_1[0].plot(t_lin,lin_opt_data_lin, '-', label="Linear opt, $SS= " + str(round(SS_data_lin_fit_lin,3)) +"$" ,color=(0/256,68/256,27/256),linewidth=3.0)
# The cubic model fitted to linear data
axs_1[0].plot(t_cube_opt_data_lin,cube_opt_data_lin, '-', label="Cube opt, $SS= " + str(round(SS_data_lin_fit_cube,3)) +"$" ,color=(103/256,0/256,31/256),linewidth=3.0)
# Add a grid as well
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
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# The cubic model
# The original cubic curve
axs_1[1].plot(t_cube,data_cube, '<', label="Data cube, $e_i\\sim\\mathcal{N}(0,\\sigma),\quad\\sigma="+ str(round(sigma_cube,3))+ "$" ,color=(0/256,0/256,0/256),linewidth=3.0)
# The linear model fitted to linear data
axs_1[1].plot(t_cube,lin_opt_data_cube, '-', label="Linear opt, $SS= " + str(round(SS_data_cube_fit_lin,3)) +"$" ,color=(0/256,68/256,27/256),linewidth=3.0)
# The cubic model fitted to linear data
axs_1[1].plot(t_cube_opt_data_cube,cube_opt_data_cube, '-', label="Cube opt, $SS= " + str(round(SS_data_cube_fit_cube,3)) +"$" ,color=(103/256,0/256,31/256),linewidth=3.0)
# Add a lovely grid
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

#=================================================================================
#=================================================================================
# Plot the symmetries in LaTeX
#=================================================================================
#=================================================================================
# Fit to data generated by the line
plot_LaTeX_2D(t_lin,data_lin,"../Figures/LaTeX_figures/fit_line_and_cube/Input/data_line.tex","only marks, mark=halfcircle*,mark size=1.5pt,color=black,","Linear data")
plot_LaTeX_2D(t_lin,lin_opt_data_lin,"../Figures/LaTeX_figures/fit_line_and_cube/Input/data_line.tex","color=lin_1,line width=2pt,","Fitted line")
plot_LaTeX_2D(t_cube_opt_data_lin,cube_opt_data_lin,"../Figures/LaTeX_figures/fit_line_and_cube/Input/data_line.tex","color=cube_1,line width=2pt,","Fitted cube")
# Fit to data generated by the cube
plot_LaTeX_2D(t_cube,data_cube,"../Figures/LaTeX_figures/fit_line_and_cube/Input/data_cube.tex","only marks, mark=halfcircle*,mark size=1.5pt,color=black,","Cubic data")
plot_LaTeX_2D(t_cube,lin_opt_data_cube,"../Figures/LaTeX_figures/fit_line_and_cube/Input/data_cube.tex","color=lin_1,line width=2pt,","Fitted line")
plot_LaTeX_2D(t_cube_opt_data_cube,cube_opt_data_cube,"../Figures/LaTeX_figures/fit_line_and_cube/Input/data_cube.tex","color=cube_1,line width=2pt,","Fitted cube")
