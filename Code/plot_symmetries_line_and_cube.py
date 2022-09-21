#=================================================================================
#=================================================================================
# Script:"plot_symmetries_line_and_cube"
# Date: 2022-09-21
# Implemented by: Johannes Borgqvist
# Description:
# The script plot the rotation symmetry of the line y(t)=Ct and the
# scaling symmetry of the cube y(t)=Ct^3
#=================================================================================
#=================================================================================
#=================================================================================
#=================================================================================
# Import Libraries
#=================================================================================
#=================================================================================
from numpy import * # For numerical calculations,
import matplotlib.pyplot as plt # For plotting,
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
#=================================================================================
#=================================================================================
# The linear model
#=================================================================================
#=================================================================================
# The time vector
t = linspace(-1,1,100)
# The arbitrary coefficient
C = 1
# The transformation parameter
eps_lin = pi/6
# Define a nice epsilon vector that goes with the transformation parameter
eps_vec_lin = linspace(0,eps_lin,num=75,endpoint=True)
# Define some magical indices that we want to transform
magical_indices_lin = [37, 40, 43, 55, 58, 61]
# The original curve of the linear model
y_lin_ori = array([C*t_temp for t_temp in t])
# Define the transformed parameter
C_hat_lin_1 = ((sin(eps_lin)+C*cos(eps_lin))/(cos(eps_lin)-C*sin(eps_lin)))
C_hat_lin_2 = ((sin(2*eps_lin)+C*cos(2*eps_lin))/(cos(2*eps_lin)-C*sin(2*eps_lin)))
# The transformed curve of the linear model
y_lin_hat = array([C_hat_lin_1*t_temp for t_temp in t])
# Allocate some memory for our rotation symmetry
rot = []
# Loop over our magical indices and start rotating
for magical_index in magical_indices_lin:
    # Extract the point we are transforming
    t_temp = t[magical_index]
    y_temp = y_lin_ori[magical_index]
    # Calculate our symmetry for the given point
    rot_temp = [array([t_temp*cos(eps_temp)-y_temp*sin(eps_temp) for eps_temp in eps_vec_lin]), array([t_temp*sin(eps_temp)+y_temp*cos(eps_temp) for eps_temp in eps_vec_lin])]
    # Append our lovely symmetry
    rot.append(rot_temp)
# Let's do a second transformation as well
# The transformed curve of the linear model
y_lin_hat_2 = array([C_hat_lin_2*t_temp for t_temp in t])
# Loop over our magical indices and start rotating
for magical_index in magical_indices_lin:
    # Extract the point we are transforming
    t_temp = t[magical_index]
    y_temp = y_lin_hat[magical_index]
    # Calculate our symmetry for the given point
    rot_temp = [array([t_temp*cos(eps_temp)-y_temp*sin(eps_temp) for eps_temp in eps_vec_lin]), array([t_temp*sin(eps_temp)+y_temp*cos(eps_temp) for eps_temp in eps_vec_lin])]
    # Append our lovely symmetry
    rot.append(rot_temp)
#=================================================================================
#=================================================================================
# The cubic model
#=================================================================================
#=================================================================================
# The transformation parameter
eps_cube = 0.5
# Define a nice epsilon vector that goes with the transformation parameter
eps_vec_cube = linspace(0,eps_cube,num=75,endpoint=True)
# Do some magical indices close to 100
magical_indices_cube = [50,60,70,130,140,150]
# Define a new time vector for the cube
t_cube = linspace(-2,2,200)
# The original curve of the cubic model
y_cube_ori = array([C*t_temp**3 for t_temp in t_cube])
# Define the transformed parameter
C_hat_cube_1 = C*exp(-2*eps_cube)
# Define the transformed curve
y_cube_hat = array([C_hat_cube_1*t_temp**3 for t_temp in t_cube])

# Allocate some memory for our rotation symmetry
scaling = []
# Loop over our magical indices and start rotating
for magical_index in magical_indices_cube:
    # Extract the point we are transforming
    t_temp = t_cube[magical_index]
    y_temp = y_cube_ori[magical_index]
    # Calculate our symmetry for the given point
    scale_temp = [array([t_temp*exp(eps_temp) for eps_temp in eps_vec_cube]), array([y_temp*exp(eps_temp) for eps_temp in eps_vec_cube])]
    # Append our lovely symmetry
    scaling.append(scale_temp)
# Let's do another transformation    
# Define the transformed parameter
C_hat_cube_2 = C*exp(-4*eps_cube)
# Define the transformed curve
y_cube_hat_2 = array([C_hat_cube_2*t_temp**3 for t_temp in t_cube])    
# Loop over our magical indices and start rotating
for magical_index in magical_indices_cube:
    # Extract the point we are transforming
    t_temp = t_cube[magical_index]
    y_temp = y_cube_hat[magical_index]
    # Calculate our symmetry for the given point
    scale_temp = [array([t_temp*exp(eps_temp) for eps_temp in eps_vec_cube]), array([y_temp*exp(eps_temp) for eps_temp in eps_vec_cube])]
    # Append our lovely symmetry
    scaling.append(scale_temp)
#=================================================================================
#=================================================================================
# Plot the symmetries
#=================================================================================
#=================================================================================
# Set all parameters to tex
plt.rcParams['text.usetex'] = True
# Define a figure window with two subfigures
fig_5, axs_5 = plt.subplots(1, 2, constrained_layout=True, figsize=(20, 8))
# The linear model
# The original solution
axs_5[0].plot(t, y_lin_ori, '-', label="$y(t)$" ,color=(0/256,68/256,27/256),linewidth=3.0)
# The transformed curve
axs_5[0].plot(t, y_lin_hat, '-', label="$\\hat{y}(t,\\epsilon=\\pi/6)$" ,color=(0/256,109/256,44/256),linewidth=3.0)
# The second transformed curve
axs_5[0].plot(t, y_lin_hat_2, '-', label="$\\hat{y}(t,2\\epsilon=\\pi/3)$" ,color=(35/256,139/256,69/256),linewidth=3.0)
# The rotation symmetry
for index in range(len(rot)):
    if index ==0:
        axs_5[0].plot(rot[index][0], rot[index][1], '--', label="$\\left.\\Gamma_{\\epsilon}^{R}\\right|_{\\epsilon=\\pi/6}$" ,color=(0/256,0/256,0/256),linewidth=3.0)
    else:
        axs_5[0].plot(rot[index][0], rot[index][1], '--',color=(0/256,0/256,0/256),linewidth=3.0)
axs_5[0].grid()
# Set the limits
axs_5[0].set_xlim([-1, 1])
axs_5[0].set_ylim([-1, 1])
# Legends and axes labels
axs_5[0].legend(loc='best',prop={"size":20})
axs_5[0].set_ylabel('Explanatory variable, $y(t)$',fontsize=25)
axs_5[0].set_xlabel('Response variable, $t$',fontsize=25)
# Change the size of the ticks
axs_5[0].tick_params(axis='both', which='major', labelsize=20)
axs_5[0].tick_params(axis='both', which='minor', labelsize=20)
# Title and saving the figure
axs_5[0].set_title("The linear model, $y(t)="+ str(C) + "\\;t$",fontsize=30,weight='bold');
# The cubic model
# The original cubic curve
axs_5[1].plot(t_cube, y_cube_ori, '-', label="$y(t)$" ,color=(103/256,0/256,31/256),linewidth=3.0)
# The transformed cubic curve
axs_5[1].plot(t_cube, y_cube_hat, '-', label="$\\hat{y}(t,\\epsilon=" + str(eps_cube)+ ")$" ,color=(152/256,0/256,67/256),linewidth=3.0)
# The second transformation
axs_5[1].plot(t_cube, y_cube_hat_2, '-', label="$\\hat{y}(t,2\\epsilon=" + str(2*eps_cube)+ ")$" ,color=(206/256,18/256,86/256),linewidth=3.0)
# The scaling symmetry
for index in range(len(scaling)):
    if index ==0:
        axs_5[1].plot(scaling[index][0], scaling[index][1], '--', label="$\\left.\\Gamma_{\\epsilon}^{S}\\right|_{\\epsilon=" + str(eps_cube)+ "}$" ,color=(0/256,0/256,0/256),linewidth=3.0)
    else:
        axs_5[1].plot(scaling[index][0], scaling[index][1], '--',color=(0/256,0/256,0/256),linewidth=3.0)
axs_5[1].grid()
# Set the limits
axs_5[1].set_xlim([-2, 2])
axs_5[1].set_ylim([-2, 2])
# Legends and axes labels
axs_5[1].legend(loc='best',prop={"size":20})
axs_5[1].set_ylabel('Explanatory variable, $y(t)$',fontsize=25)
axs_5[1].set_xlabel('Response variable, $t$',fontsize=25)
# Change the size of the ticks
axs_5[1].tick_params(axis='both', which='major', labelsize=20)
axs_5[1].tick_params(axis='both', which='minor', labelsize=20)
# Title and saving the figure
axs_5[1].set_title("The cubic model, $y(t)="+ str(C) + "\\;t^3$",fontsize=30,weight='bold');
# Save the figure
fig_5.savefig('../Figures/symmetries_line_and_cube.png')
# Show the figure
plt.show()


#=================================================================================
#=================================================================================
# Plot the symmetries in LaTeX
#=================================================================================
#=================================================================================
# LINEAR MODEL
# The original solution of the linear model
plot_LaTeX_2D(t, y_lin_ori,"../Figures/symmetries_line_and_cube/Input/line.tex","color=lin_1,line width=2pt,","$y(t)$")
# The first transformed solution of the linear model
plot_LaTeX_2D(t, y_lin_hat,"../Figures/symmetries_line_and_cube/Input/line.tex","color=lin_2,line width=2pt,","$\\hat{y}(t,\\epsilon)$")
# The second transformed solution of the linear model
plot_LaTeX_2D(t, y_lin_hat_2,"../Figures/symmetries_line_and_cube/Input/line.tex","color=lin_3,line width=2pt,","$\\hat{y}(t,2\\epsilon)$")
# The rotation symmetry
for index in range(len(rot)):
    if index ==0:
        plot_LaTeX_2D(rot[index][0], rot[index][1],"../Figures/symmetries_line_and_cube/Input/line.tex","color=black,->,>=latex,densely dashed","$\\Gamma_{\\epsilon}^{R}$")
    else:
        plot_LaTeX_2D(rot[index][0], rot[index][1],"../Figures/symmetries_line_and_cube/Input/line.tex","color=black,->,>=latex,densely dashed",[])
# CUBIC MODEL
# The original solution of the linear model
plot_LaTeX_2D(t_cube, y_cube_ori,"../Figures/symmetries_line_and_cube/Input/cube.tex","color=cube_1,line width=2pt,","$y(t)$")
# The first transformed solution of the linear model
plot_LaTeX_2D(t_cube, y_cube_hat,"../Figures/symmetries_line_and_cube/Input/cube.tex","color=cube_2,line width=2pt,","$\\hat{y}(t,\\epsilon)$")
# The second transformed solution of the linear model
plot_LaTeX_2D(t_cube, y_cube_hat_2,"../Figures/symmetries_line_and_cube/Input/cube.tex","color=cube_3,line width=2pt,","$\\hat{y}(t,2\\epsilon)$")
# The rotation symmetry
for index in range(len(scaling)):
    if index ==0:
        plot_LaTeX_2D(scaling[index][0], scaling[index][1],"../Figures/symmetries_line_and_cube/Input/cube.tex","color=black,->,>=latex,densely dashed","$\\Gamma_{\\epsilon}^{S}$")
    else:
        plot_LaTeX_2D(scaling[index][0], scaling[index][1],"../Figures/symmetries_line_and_cube/Input/cube.tex","color=black,->,>=latex,densely dashed",[])
