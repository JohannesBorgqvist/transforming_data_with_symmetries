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
from scipy import integrate # For solving ODEs
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
# Function 2: circular_model
def circular_model(x, k):
    return sqrt(k-x**2)
# Function 3: parabolic_model
def parabolic_model(x, k1, k2):
    return k1*x**2+k2
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
# Function 5: theoretical_error_circular.
# The function returns the predicted theoretical
# error for the roation operator associated with
# the linear model. It takes 4 input arguments:
# 1. res: the previous residual,
# 2. epsilon: the transformation parameter,
# 3. t: the time,
# 4. y: the y-value at time t.
def theoretical_error_circular(res,epsilon,t,y):
    # Transformed
    
    # Calculate the complicated term
    complicated_term = (epsilon/t)*(2*y+res)
    # Calculate the factor based on this
    factor = (1-complicated_term)
    # Add the squared term
    factor += -((epsilon**2)/(2))*((3*y**2)/(t**2))
    factor += ((epsilon**2)/(2))*2
    factor += -((epsilon**2)/(2))*((3*y)/(t**2))*res
    factor += -((epsilon**2)/(2))*((res**2)/(2*t**2))    
    # Finally we can return the predicted theoretical error
    return (res**2)*(factor**2)
# Function 5: theoretical_error_cubic.
# The function returns the predicted theoretical
# error for the roation operator associated with
# the linear model. It takes 4 input arguments:
# 1. res: the previous residual,
# 2. epsilon: the transformation parameter,
# 3. t: the time,
# 4. y: the y-value at time t.
def theoretical_error_cubic(res,epsilon,t,y):
    # Define the linear term
    factor = (1+4*epsilon)
    # Add the squared term
    factor += ((epsilon**2)/(2)) * (4+6*(y/t)+(3/t)*res)
    # Finally we can return the predicted theoretical error
    return (res**2)*(factor**2)
#=================================================================================
#=================================================================================
# Overall properties of the linear model and the cube
#=================================================================================
#=================================================================================
# Define our constant C
C_circle = 1
C_p1 = -1
C_p2 = 1
# Define the noise level
sigma_circle = 0.1
sigma_parabola = 0.1
# Define two transformation parameters
# for our two models
epsilon_circle = 0.5
epsilon_parabola = 0.5
# Define the ranges for the time
t_min = 0
t_max = 1
#=================================================================================
#=================================================================================
# Generate data
# UNCOMMENT THIS SECTION IF YOU WANT TO GENERATE NEW DATA
#=================================================================================
#=================================================================================
# Define the number of data points
num_points = 30
# Define the errors of the circle model
errors_circle = random.normal(0, sigma_circle, size=(num_points,))
# Define the errors of the parabolic model
errors_parabola = random.normal(0, sigma_parabola, size=(num_points,))
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# Generate data from the linear model
t_circle = linspace(t_min,t_max,num_points,endpoint=True)
data_circle = array([sqrt(C_circle-t_temp**2)+errors_circle[index] for index,t_temp in enumerate(t_circle)])
# Define the file name
file_name_circle = "../Data/new_data_circle_data_sigma_" + str(round(sigma_circle,3)).replace(".","p") + ".csv"
# Save the linear data to file
pd.DataFrame(data = array([list(t_circle), list(data_circle)]).T,index=["Row "+str(temp_index+1) for temp_index in range(num_points)],columns=["t", "y(t)"]).to_csv(file_name_circle)
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# Generate data from the cubic model
t_parabola = linspace(t_min,t_max,num_points,endpoint=True)
data_parabola = array([C_p1*(t_temp)**3+C_p2+errors_parabola[index] for index,t_temp in enumerate(t_parabola)])
# Define the file name
file_name_parabola = "../Data/new_data_parabola_data_sigma_" + str(round(sigma_parabola,3)).replace(".","p") + ".csv"
# Save the cubic data to file
pd.DataFrame(data = array([list(t_parabola), list(data_parabola)]).T,index=["Row "+str(temp_index+1) for temp_index in range(num_points)],columns=["t", "y(t)"]).to_csv(file_name_parabola)
#=================================================================================
#=================================================================================
# Read data from a file
#=================================================================================
#=================================================================================
# # Linear data: define the file we want to read the data from
# cubic_data_str = "../Data/cubic_data_sigma_" + str(round(sigma_cube,3)).replace(".","p") + ".csv"
# # Read the data from the file
# t_cube, data_cube = read_generated_data(cubic_data_str)
# # Cubic data: define the file we want to read the data from
# linear_data_str = "../Data/linear_data_sigma_" + str(sigma_lin).replace(".","p") + ".csv"
# # Read the data from the file
# t_lin, data_lin = read_generated_data(linear_data_str)
# # Calculate the number of points
# num_points = len(t_lin)
# #=================================================================================
# #=================================================================================
# # Fit models to data
# #=================================================================================
# #=================================================================================
# #---------------------------------------------------------------------------------
# #---------------------------------------------------------------------------------
# # LINEAR DATA
# #---------------------------------------------------------------------------------
# #---------------------------------------------------------------------------------
# # 1. Fit the linear model to linear data
popt_data_circle_fit_circle, pcov_data_circle_fit_circle = curve_fit(circular_model, t_circle, data_circle)
# # Calculate residuals
res_data_circle_fit_circle = circular_model(t_circle,*popt_data_circle_fit_circle)-data_circle
# # Calculate the sum of squares, SS
SS_data_circle_fit_circle = sum(array([res**2 for res in res_data_circle_fit_circle]))/(len(t_circle)-1)
# # Extract parameter
C_circle_opt_data_circle = popt_data_circle_fit_circle[0]
# # Generate the optimal line
t_circle_plot = linspace(t_circle[0],t_circle[-1],100)
circle_opt_data_circle = array([sqrt(C_circle_opt_data_circle-t_temp**2) for t_temp in t_circle_plot])
# #---------------------------------------------------------------------------------
# # 2. Fit the cubic model to linear data
popt_data_circle_fit_parabola, pcov_data_circle_fit_parabola = curve_fit(parabolic_model, t_circle, data_circle)
# # Calculate residuals
res_data_circle_fit_parabola = parabolic_model(t_circle,*popt_data_circle_fit_parabola)-data_circle
# # Calculate the sum of squares, SS
SS_data_circle_fit_parabola = sum(array([res**2 for res in res_data_circle_fit_parabola]))/(len(t_circle)-1)
# # Extract parameter
C_p1_opt_data_circle = popt_data_circle_fit_parabola[0]
C_p2_opt_data_circle = popt_data_circle_fit_parabola[1]
# # Generate the optimal cube
t_circle_opt_data_parabola = linspace(t_circle[0],t_circle[-1],100)
parabola_opt_data_circle = array([C_p1_opt_data_circle*(t_temp)**2+C_p2_opt_data_circle for t_temp in t_circle_opt_data_parabola])
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# PARABOLIC DATA
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# # 1. Fit the linear model to linear data
popt_data_parabola_fit_circle, pcov_data_parabola_fit_circle = curve_fit(circular_model, t_parabola, data_parabola)
# # Calculate residuals
res_data_parabola_fit_circle = circular_model(t_parabola,*popt_data_parabola_fit_circle)-data_parabola
# # Calculate the sum of squares, SS
SS_data_parabola_fit_circle = sum(array([res**2 for res in res_data_parabola_fit_circle]))/(len(t_parabola)-1)
# # Extract parameter
C_circle_opt_data_parabola = popt_data_parabola_fit_circle[0]
# # Generate the optimal line
t_parabola_plot = linspace(t_parabola[0],t_parabola[-1],100)
circle_opt_data_parabola = array([sqrt(C_circle_opt_data_parabola-t_temp**2) for t_temp in t_parabola_plot])
# #---------------------------------------------------------------------------------
# # 2. Fit the cubic model to linear data
popt_data_parabola_fit_parabola, pcov_data_parabola_fit_parabola = curve_fit(parabolic_model, t_parabola, data_parabola)
# # Calculate residuals
res_data_parabola_fit_parabola = parabolic_model(t_parabola,*popt_data_parabola_fit_parabola)-data_parabola
# # Calculate the sum of squares, SS
SS_data_parabola_fit_parabola = sum(array([res**2 for res in res_data_parabola_fit_parabola]))/(len(t_parabola)-1)
# # Extract parameter
C_p1_opt_data_parabola = popt_data_parabola_fit_parabola[0]
C_p2_opt_data_parabola = popt_data_parabola_fit_parabola[1]
# # Generate the optimal cube
t_parabola_opt_data_parabola = linspace(t_parabola[0],t_parabola[-1],100)
parabola_opt_data_parabola = array([C_p1_opt_data_parabola*(t_temp)**2+C_p2_opt_data_parabola for t_temp in t_parabola_opt_data_parabola])
#=================================================================================
#=================================================================================
# Plot the data as well as the models when we fit them to data
#=================================================================================
#=================================================================================
# Set all parameters to tex
plt.rcParams['text.usetex'] = True
# Define a figure window with two subfigures
fig_1, axs_1 = plt.subplots(1, 2, constrained_layout=True, figsize=(20, 8))
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# The linear data
axs_1[0].plot(t_circle,data_circle, 'o', label="Data circle, $e_i\\sim\\mathcal{N}(0,\\sigma),\quad\\sigma="+ str(round(sigma_circle,3))+ "$" ,color=(0/256,0/256,0/256),linewidth=3.0)
# The linear model fitted to linear data
axs_1[0].plot(t_circle_plot,circle_opt_data_circle, '-', label="Circle opt, $SS= " + str(round(SS_data_circle_fit_circle,3)) +"$" ,color=(0/256,68/256,27/256),linewidth=3.0)
# The cubic model fitted to linear data
axs_1[0].plot(t_circle_opt_data_parabola,parabola_opt_data_circle, '-', label="Parabola opt, $SS= " + str(round(SS_data_circle_fit_parabola,3)) +"$" ,color=(103/256,0/256,31/256),linewidth=3.0)
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
axs_1[0].set_title("Data generated by the linear model, $y(t)=\\sqrt{"+ str(C_circle) + "-t^2}$",fontsize=30,weight='bold');
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# The cubic model
# The original cubic curve
axs_1[1].plot(t_parabola,data_parabola, '<', label="Data parabola, $e_i\\sim\\mathcal{N}(0,\\sigma),\quad\\sigma="+ str(round(sigma_parabola,3))+ "$" ,color=(0/256,0/256,0/256),linewidth=3.0)
# The linear model fitted to linear data
axs_1[1].plot(t_parabola_plot,circle_opt_data_parabola, '-', label="Circle opt, $SS= " + str(round(SS_data_parabola_fit_circle,3)) +"$" ,color=(0/256,68/256,27/256),linewidth=3.0)
# The cubic model fitted to linear data
axs_1[1].plot(t_parabola_opt_data_parabola,parabola_opt_data_parabola, '-', label="Parabola opt, $SS= " + str(round(SS_data_parabola_fit_parabola,3)) +"$" ,color=(103/256,0/256,31/256),linewidth=3.0)
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
axs_1[1].set_title("Data generated by the cubic model, $y(t)="+ str(C_p1) + "t^2+" + str(C_p2)+ "$",fontsize=30,weight='bold');
# Save the figure
#fig_1.savefig('../Figures/data_line_and_cube.png')
# Show the figure
#plt.show()

# #=================================================================================
# #=================================================================================
# # Plot the symmetries in LaTeX
# #=================================================================================
# #=================================================================================
# # Fit to data generated by the line
# plot_LaTeX_2D(t_lin,data_lin,"../Figures/LaTeX_figures/fit_line_and_cube/Input/data_line.tex","only marks, mark=halfcircle*,mark size=1.5pt,color=black,","Linear data")
# plot_LaTeX_2D(t_lin,lin_opt_data_lin,"../Figures/LaTeX_figures/fit_line_and_cube/Input/data_line.tex","color=lin_1,line width=2pt,","Fitted line")
# plot_LaTeX_2D(t_cube_opt_data_lin,cube_opt_data_lin,"../Figures/LaTeX_figures/fit_line_and_cube/Input/data_line.tex","color=cube_1,line width=2pt,","Fitted cube")
# # Fit to data generated by the cube
# plot_LaTeX_2D(t_cube,data_cube,"../Figures/LaTeX_figures/fit_line_and_cube/Input/data_cube.tex","only marks, mark=halfcircle*,mark size=1.5pt,color=black,","Cubic data")
# plot_LaTeX_2D(t_cube,lin_opt_data_cube,"../Figures/LaTeX_figures/fit_line_and_cube/Input/data_cube.tex","color=lin_1,line width=2pt,","Fitted line")
# plot_LaTeX_2D(t_cube_opt_data_cube,cube_opt_data_cube,"../Figures/LaTeX_figures/fit_line_and_cube/Input/data_cube.tex","color=cube_1,line width=2pt,","Fitted cube")
# #=================================================================================
# #=================================================================================
# # Fit models to transformed data
# #=================================================================================
# #=================================================================================
# #---------------------------------------------------------------------------------
# #---------------------------------------------------------------------------------
# # LINEAR DATA
# #---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# LINEAR MODEL FITTED TO LINEAR DATA
#---------------------------------------------------------------------------------
# Define a vector with transformation parameters that we wish to transform the
# data with
epsilon_vec_circle = linspace(0,epsilon_circle,100,endpoint=True)
# Allocate memory for the sum of squares corresponding to the fit of the linear
# model to the transformed linear data
SS_data_circle_fit_circle_transf = []
# Also, allocate memory for the theoretical prediction
SS_theo_data_circle_fit_circle = []
# Loop over the transformation parameters, transform the data and fit the model
# to the transformed data
for epsilon in epsilon_vec_circle:
    # Transform the time points with the given transformation parameter
    t_trans = array([t_circle[index]*cos(epsilon)-y*sin(epsilon) for index,y in enumerate(data_circle)])
    # Transform the data points with the given transformation parameter
    y_trans = array([t_circle[index]*sin(epsilon)+y*cos(epsilon) for index,y in enumerate(data_circle)])
    # Extract the temporary parameter
    C_temp = popt_data_circle_fit_circle[0]
    # Calculate the transformed parameter C in y=Ct
    C_hat = C_temp
    # Next, we calculate the residuals
    residuals_y = array([sqrt(C_hat-t_temp**2) for t_temp in t_trans])-y_trans
    #residuals_y = circle_opt_data_circle-y_trans
    residuals_t = t_trans - t_circle
    residuals = array([sqrt(residuals_y[index]**2 + residuals_t[index]**2) for index,y_temp in enumerate(residuals_y)])
    # Lastly, we append the sum of squares
    SS_data_circle_fit_circle_transf.append(sum(array([res**2 for res in residuals]))/(len(t_trans)-1))
    # Now, let's calculate the theoretical prediction
    #SS_theo_data_circle_fit_circle.append(sum(array([theoretical_error_circular(res,epsilon,t_circle[index],data_circle[index]) for index,res in enumerate(res_data_circle_fit_circle) if index > 0]))/(len(t_circle)-1))
    SS_theo_data_circle_fit_circle.append(SS_data_circle_fit_circle)
# Lastly, we cast our list as an np.array    
SS_data_circle_fit_circle = array(SS_data_circle_fit_circle)
# Also, we cast or theoretical list as an array
SS_theo_data_circle_fit_circle = array(SS_theo_data_circle_fit_circle)
# #---------------------------------------------------------------------------------
# # CUBIC MODEL FITTED TO LINEAR DATA
# #---------------------------------------------------------------------------------
# # Define a vector with transformation parameters that we wish to transform the
# # data with
# epsilon_vec_cube = linspace(0,epsilon_cube,100,endpoint=True)
# # Allocate memory for the sum of squares corresponding to the fit of the linear
# # model to the transformed linear data
# SS_data_lin_fit_cube = []
# # Also, allocate memory for the theoretical prediction
# SS_theo_data_lin_fit_cube = []
# # Loop over the transformation parameters, transform the data and fit the model
# # to the transformed data
# for epsilon in epsilon_vec_cube:
#     # Transform the time points with the given transformation parameter
#     t_trans = array([t_lin[index]*exp(epsilon) for index,y in enumerate(data_lin)])
#     # Transform the data points with the given transformation parameter
#     y_trans = array([y*exp(epsilon) for index,y in enumerate(data_lin)])
#     # Extract the temporary parameter
#     C_temp = popt_data_lin_fit_cube[0]
#     # Calculate the transformed parameter C in y=Ct^3
#     C_hat = C_temp*exp(-2*epsilon)
#     # Next, we calculate the residuals
#     residuals = array([C_hat*(t_temp**3) for t_temp in t_trans])-y_trans
#     # Lastly, we append the sum of squares
#     SS_data_lin_fit_cube.append(sum(array([res**2 for res in residuals]))/(len(t_trans)-1))
#     # Also, we calculate our theoretical prediction
#     SS_theo_data_lin_fit_cube.append(sum(array([theoretical_error_cubic(res,epsilon,t_lin[index],data_lin[index]) for index,res in enumerate(res_data_lin_fit_cube)]))/(len(t_trans)-1))
# # Lastly, we cast our list as an np.array    
# SS_data_lin_fit_cube = array(SS_data_lin_fit_cube)
# # Also, we cast our sum of squares as an np array
# SS_theo_data_lin_fit_cube = array(SS_theo_data_lin_fit_cube)
# #---------------------------------------------------------------------------------
# #---------------------------------------------------------------------------------
# # CUBIC DATA
# #---------------------------------------------------------------------------------
# #---------------------------------------------------------------------------------
# # LINEAR MODEL FITTED TO CUBIC DATA
# #---------------------------------------------------------------------------------
# # Allocate memory for the sum of squares corresponding to the fit of the linear
# # model to the transformed linear data
# SS_data_cube_fit_lin = []
# # Also, allocate memory for the theoretical prediction
# SS_theo_data_cube_fit_lin = []
# # Loop over the transformation parameters, transform the data and fit the model
# # to the transformed data
# for epsilon in epsilon_vec_lin:
#     # Transform the time points with the given transformation parameter
#     t_trans = array([t_cube[index]*cos(epsilon)-y*sin(epsilon) for index,y in enumerate(data_cube)])
#     # Transform the data points with the given transformation parameter
#     y_trans = array([t_cube[index]*sin(epsilon)+y*cos(epsilon) for index,y in enumerate(data_cube)])
#     # Extract the temporary parameter
#     C_temp = popt_data_cube_fit_lin[0]
#     # Calculate the transformed parameter C in y=Ct
#     C_hat = ((sin(epsilon)+C_temp*cos(epsilon))/(cos(epsilon)-C_temp*sin(epsilon)))
#     # Next, we calculate the residuals
#     residuals = array([C_hat*t_temp for t_temp in t_trans])-y_trans
#     # Lastly, we append the sum of squares
#     SS_data_cube_fit_lin.append(sum(array([res**2 for res in residuals]))/(len(t_trans)-1))
#     # Now, let's calculate the theoretical prediction
#     SS_theo_data_cube_fit_lin.append(sum(array([theoretical_error_linear(res,epsilon,t_cube[index],data_cube[index]) for index,res in enumerate(res_data_cube_fit_lin)]))/(len(t_lin)-1))    
# # Lastly, we cast our list as an np.array    
# SS_data_cube_fit_lin = array(SS_data_cube_fit_lin)
# # Next, we cast our list as an np.array
# SS_theo_data_cube_fit_lin = array(SS_theo_data_cube_fit_lin)
# #---------------------------------------------------------------------------------
# # CUBIC MODEL FITTED TO CUBIC DATA
# #---------------------------------------------------------------------------------
# # Allocate memory for the sum of squares corresponding to the fit of the linear
# # model to the transformed linear data
# SS_data_cube_fit_cube = []
# # Also, allocate memory for the theoretical prediction
# SS_theo_data_cube_fit_cube = []
# # Loop over the transformation parameters, transform the data and fit the model
# # to the transformed data
# for epsilon in epsilon_vec_cube:
#     # Transform the time points with the given transformation parameter
#     t_trans = array([t_cube[index]*exp(epsilon) for index,y in enumerate(data_cube)])
#     # Transform the data points with the given transformation parameter
#     y_trans = array([y*exp(epsilon) for index,y in enumerate(data_cube)])
#     # Extract the temporary parameter
#     C_temp = popt_data_cube_fit_cube[0]
#     # Calculate the transformed parameter C in y=Ct^3
#     C_hat = C_temp*exp(-2*epsilon)
#     # Next, we calculate the residuals
#     residuals = array([C_hat*(t_temp**3) for t_temp in t_trans])-y_trans
#     # Lastly, we append the sum of squares
#     SS_data_cube_fit_cube.append(sum(array([res**2 for res in residuals]))/(len(t_trans)-1))
#     # Also, we calculate our theoretical prediction
#     SS_theo_data_cube_fit_cube.append(sum(array([theoretical_error_cubic(res,epsilon,t_cube[index],data_cube[index]) for index,res in enumerate(res_data_cube_fit_cube)]))/(len(t_trans)-1))    
# # Lastly, we cast our list as an np.array    
# SS_data_cube_fit_cube = array(SS_data_cube_fit_cube)
# # We also cast it to an np.array
# SS_theo_data_cube_fit_cube = array(SS_theo_data_cube_fit_cube)
# #=================================================================================
# #=================================================================================
# # Plot the data as well as the models when we fit them to data
# #=================================================================================
# #=================================================================================
# # Set all parameters to tex
plt.rcParams['text.usetex'] = True
# # Define a figure window with two subfigures
fig_2, axs_2 = plt.subplots(1, 2, constrained_layout=True, figsize=(20, 8))
# #---------------------------------------------------------------------------------
# #---------------------------------------------------------------------------------
# # The linear model
# # The linear data: fitted line
axs_2[0].plot(epsilon_vec_circle,SS_data_circle_fit_circle_transf, '-', label="Fitted circle" ,color=(0/256,68/256,27/256),linewidth=3.0)
# # The theoretical prediction of the linear model
axs_2[0].plot(epsilon_vec_circle,SS_theo_data_circle_fit_circle, 'o', label="Theoretical circle" ,color=(0/256,109/256,44/256),linewidth=15.0)
# # The linear data: fitted cube
# axs_2[0].plot(epsilon_vec_cube,SS_data_lin_fit_cube, '-', label="Fitted cube" ,color=(103/256,0/256,31/256),linewidth=3.0)
# # The theoretical prediction of the cubic model
# axs_2[0].plot(epsilon_vec_lin,SS_theo_data_lin_fit_cube, 'D', label="Theoretical cube" ,color=(152/256,0/256,67/256),linewidth=5.0)
# Add a grid as well
axs_2[0].grid()
# Set the limits
#axs_2[0].set_xlim([0, 0.5])
#axs_2[0].set_ylim([0, 0.3])
# Legends and axes labels
axs_2[0].legend(loc='best',prop={"size":20})
axs_2[0].set_ylabel('Fit, $SS(\\epsilon)$',fontsize=25)
axs_2[0].set_xlabel('Transformation parameter, $\\epsilon$',fontsize=25)
# Change the size of the ticks
axs_2[0].tick_params(axis='both', which='major', labelsize=20)
axs_2[0].tick_params(axis='both', which='minor', labelsize=20)
# Title and saving the figure
axs_2[0].set_title("Data generated by the linear model, $y(t)=\\sqrt{"+ str(C_circle) + "-t^2}$",fontsize=30,weight='bold');
# #---------------------------------------------------------------------------------
# #---------------------------------------------------------------------------------
# # The cubic model
# # The cubic data: fitted line
# axs_2[1].plot(epsilon_vec_lin,SS_data_cube_fit_lin, '-', label="Fitted line" ,color=(0/256,68/256,27/256),linewidth=3.0)
# # The theoretical prediction of the linear model
# axs_2[1].plot(epsilon_vec_lin,SS_theo_data_cube_fit_lin, 'o', label="Theoretical line" ,color=(0/256,109/256,44/256),linewidth=5.0)
# # The cubic data: fitted cube
# axs_2[1].plot(epsilon_vec_cube,SS_data_cube_fit_cube, '-', label="Fitted cube" ,color=(103/256,0/256,31/256),linewidth=3.0)
# # The theoretical prediction of the cubic model
# axs_2[1].plot(epsilon_vec_lin,SS_theo_data_cube_fit_cube, 'D', label="Theoretical cube" ,color=(152/256,0/256,67/256),linewidth=5.0)
# # Add a lovely grid
# axs_2[1].grid()
# # Set the limits
# #axs_2[1].set_xlim([-2, 2])
# #axs_2[1].set_ylim([0, 0.3])
# # Legends and axes labels
# axs_2[1].legend(loc='best',prop={"size":20})
# axs_2[1].set_ylabel('Fit, $SS(\\epsilon)$',fontsize=25)
# axs_2[1].set_xlabel('Transformation parameter, $\\epsilon$',fontsize=25)
# # Change the size of the ticks
# axs_2[1].tick_params(axis='both', which='major', labelsize=20)
# axs_2[1].tick_params(axis='both', which='minor', labelsize=20)
# # Title and saving the figure
# axs_2[1].set_title("Data generated by the cubic model, $y(t)="+ str(C_cube) + "\\;t^3$",fontsize=30,weight='bold');
# # Save the figure
# fig_2.savefig('../Figures/transf_data_line_and_cube.png')
# # Show the figure
plt.show()
# #=================================================================================
# #=================================================================================
# # Plot the fit to transformed data in LaTeX
# #=================================================================================
# #=================================================================================
# # Fit to data generated by the line
# plot_LaTeX_2D(epsilon_vec_lin,SS_data_lin_fit_lin,"../Figures/LaTeX_figures/transf_data_line_and_cube/Input/data_line.tex","color=lin_1,line width=1pt,","Fitted line") # Linear fitted to data
# plot_LaTeX_2D(epsilon_vec_lin,SS_theo_data_lin_fit_lin,"../Figures/LaTeX_figures/transf_data_line_and_cube/Input/data_line.tex","color=lin_3,line width=1pt,dashdotted, every mark/.append style={solid, fill=lin_3},mark=otimes*,mark size=1.5pt","Theoretical line") # Theoretical line
# plot_LaTeX_2D(epsilon_vec_cube,SS_data_lin_fit_cube,"../Figures/LaTeX_figures/transf_data_line_and_cube/Input/data_line.tex","color=cube_1,line width=1pt,","Fitted cube") # Cubic fitted to data
# plot_LaTeX_2D(epsilon_vec_lin,SS_theo_data_lin_fit_cube,"../Figures/LaTeX_figures/transf_data_line_and_cube/Input/data_line.tex","color=cube_3,line width=1pt,densely dashdotted,every mark/.append style={solid, fill=cube_3},mark=diamond*,mark size=1.5pt","Theoretical cube") # Theoretical cube
# # Fit to data generated by the cube
# plot_LaTeX_2D(epsilon_vec_lin,SS_data_cube_fit_lin,"../Figures/LaTeX_figures/transf_data_line_and_cube/Input/data_cube.tex","color=lin_1,line width=1pt,","Fitted line") # Fitted line
# plot_LaTeX_2D(epsilon_vec_lin,SS_theo_data_cube_fit_lin,"../Figures/LaTeX_figures/transf_data_line_and_cube/Input/data_cube.tex","color=lin_3,line width=1pt,dashdotted, every mark/.append style={solid, fill=lin_3},mark=otimes*,mark size=1.5pt","Theoretical line") # Theoretical line
# plot_LaTeX_2D(epsilon_vec_cube,SS_data_cube_fit_cube,"../Figures/LaTeX_figures/transf_data_line_and_cube/Input/data_cube.tex","color=cube_1,line width=1pt,","Fitted cube") # Fitted cube
# plot_LaTeX_2D(epsilon_vec_lin,SS_theo_data_cube_fit_cube,"../Figures/LaTeX_figures/transf_data_line_and_cube/Input/data_cube.tex","color=cube_3,line width=1pt,densely dashdotted,every mark/.append style={solid, fill=cube_3},mark=diamond*,mark size=1.5pt","Theoretical cube") # Theoretical cube
