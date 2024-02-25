import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_circle_vertices(radius, num_segments, z_height):
    angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    return [[radius * np.cos(angle), radius * np.sin(angle), z_height] for angle in angles]

def plot_connector_cylinders(cylinder_radius, cylinder_length, pin_length, male_radius, female_radius, num_segments):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Male Connector
    # Ring
    male_ring_vertices = generate_circle_vertices(cylinder_radius, num_segments, cylinder_length/2)
    ax.scatter(*zip(*male_ring_vertices), c='b')  # Blue for male ring

    # Pin base
    male_pin_vertices = generate_circle_vertices(male_radius, num_segments, cylinder_length/2)
    ax.scatter(*zip(*male_pin_vertices), c='b')  # Blue for male pin
    
    # Pin
    male_pin_vertices = generate_circle_vertices(male_radius, num_segments, cylinder_length/2 - pin_length)
    ax.scatter(*zip(*male_pin_vertices), c='b')  # Blue for male pin

    # Female Connector
    # Ring
    female_ring_vertices = generate_circle_vertices(cylinder_radius, num_segments, cylinder_length/2)
    ax.scatter(*zip(*female_ring_vertices), c='r')  # Red for female ring

    # Recess base
    female_recess_vertices = generate_circle_vertices(female_radius, num_segments,cylinder_length/2 )
    ax.scatter(*zip(*female_recess_vertices), c='r')  # Red for female recess
    
    # Recess
    female_recess_vertices = generate_circle_vertices(female_radius, num_segments, cylinder_length/2 - pin_length - magnet_height)
    ax.scatter(*zip(*female_recess_vertices), c='r')  # Red for female recess

    # Main Cylinder
    # Start circle
    start_circle_vertices = generate_circle_vertices(cylinder_radius, num_segments, 0)
    ax.scatter(*zip(*start_circle_vertices), c='gray')  # Gray for cylinder start
    
    # End circle
    end_circle_vertices = generate_circle_vertices(cylinder_radius, num_segments, cylinder_length)
    ax.scatter(*zip(*end_circle_vertices), c='gray')  # Gray for cylinder end

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.title("Cylinder with Male and Female Connectors")
    plt.show()

# Parameters for the cylinder and connectors
cylinder_radius = 3.0  # Radius of the main cylinder
cylinder_length = 20.0  # Length of the main cylinder
pin_length = 5.0  # Length of the pin and recess
magnet_height = 3.0  # Height of the magnet
male_radius = 1.6  # Radius of the male pin
female_radius = 1.8  # Radius of the female recess (slightly larger than the male pin)
num_segments = 32  # Number of segments to approximate circles

plot_connector_cylinders(cylinder_radius, cylinder_length, pin_length, male_radius, female_radius, num_segments)
