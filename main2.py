import tkinter as tk
from PIL import Image, ImageTk
import threading
import random
import time
import math
import csv
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from Vector import Vector
from PIL import Image
            
        
class TrackPoint:
    def __init__(self, pos, front, left, up):
        self.pos = Vector(pos)    # Position vector
        self.front = Vector(front)  # Front axis vector
        self.left = Vector(left)    # Left axis vector
        self.up = Vector(up)      # Up axis vector
        self.circle_vertices = []

    def rotation_matrix(self):
        front_normalized = self.front.normalized
        up_normalized = self.up.normalized

        # Calculate the third vector using cross product to form an orthogonal basis
        right = Vector(-self.left.x, -self.left.y, -self.left.z)

        # Rotation matrix with 'right', 'up_normalized', and 'front_normalized' as its columns
        rot_matrix = np.array([right.components, 
                               up_normalized.components, 
                               front_normalized.components]).T
        return rot_matrix

    def generate_cylinder_section(self, radius, height, num_segments):
        angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        # Use NumPy for vectorized computation
        up_vectors = np.array(self.up.components)[:, np.newaxis]
        rotated_vectors = up_vectors * np.array([np.cos(angles), np.sin(angles), np.zeros(num_segments)])
        top_vertices = np.array(self.pos.components) + rotated_vectors.T * radius + np.array(self.front.components) * height / 2
        bottom_vertices = np.array(self.pos.components) + rotated_vectors.T * radius - np.array(self.front.components) * height / 2
        cylinder_vertices = np.vstack([top_vertices, bottom_vertices])
        return cylinder_vertices.reshape(-1).tolist() 
  
    
    def rotate_vector(self, vec, axis, angle):
        k = axis.normalized  # Ensure the axis is normalized
        vec = Vector(vec)
        # Apply Rodrigues' rotation formula
        rotated_vec = vec * math.cos(angle) + Vector.Cross(k, vec) * math.sin(angle) + k * (k * vec * (1 - math.cos(angle)))
        return rotated_vec   
    
    def compute_rotation_matrix(self, axis, angle):
        k = np.array(axis.normalized)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)

        kx, ky, kz = k
        kx2, ky2, kz2 = kx * kx, ky * ky, kz * kz
        kxky, kxkz, kykz = kx * ky, kx * kz, ky * kz
        kx_sin, ky_sin, kz_sin = kx * sin_theta, ky * sin_theta, kz * sin_theta

        # Rodrigues' rotation formula
        rotation_matrix = np.array([
            [cos_theta + kx2 * (1 - cos_theta), kxky * (1 - cos_theta) - kz_sin, kxkz * (1 - cos_theta) + ky_sin],
            [kxky * (1 - cos_theta) + kz_sin, cos_theta + ky2 * (1 - cos_theta), kykz * (1 - cos_theta) - kx_sin],
            [kxkz * (1 - cos_theta) - ky_sin, kykz * (1 - cos_theta) + kx_sin, cos_theta + kz2 * (1 - cos_theta)]
        ])

        return rotation_matrix     

class Track:
    def __init__(self):
        self.track_points = []
        self.plane_normals = []
        self.track_positions = []
        self.track_normals = []

    def generate_cylinders(self, radius, num_segments):
        angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        sin_angles = np.sin(angles)
        cos_angles = np.cos(angles)

        # Base circle vertices in the xy-plane
        base_circle = np.column_stack((radius * cos_angles, radius * sin_angles, np.zeros(num_segments)))

        all_cylinder_vertices = []
        for i in range(len(self.track_points) - 1):
            tp1 = self.track_points[i]
            tp2 = self.track_points[i + 1]

            # Rotate and translate base circle vertices for tp1 and tp2
            top_vertices = np.dot(base_circle, tp1.rotation_matrix().T) + tp1.pos.components
            bottom_vertices = np.dot(base_circle, tp2.rotation_matrix().T) + tp2.pos.components

            # Interleave top and bottom vertices
            cylinder_segment_vertices = np.empty((num_segments * 2, 3), dtype=np.float32)
            cylinder_segment_vertices[0::2] = top_vertices
            cylinder_segment_vertices[1::2] = bottom_vertices

            all_cylinder_vertices.extend(cylinder_segment_vertices)

        return all_cylinder_vertices

    def generate_and_bind_cylinder_vbo(self, radius, num_segments):
        # Generate all cylinder vertices using optimized method
        cylinder_vertices = self.generate_cylinders(radius, num_segments)

        # Convert to OpenGL-compatible format
        cylinder_vertex_data = np.array(cylinder_vertices, dtype=np.float32)

        # Generate and bind a VBO
        cylinder_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, cylinder_vbo)
        glBufferData(GL_ARRAY_BUFFER, cylinder_vertex_data.nbytes, cylinder_vertex_data, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        return cylinder_vbo

    def draw_cylinders(self, cylinder_vbo, num_vertices_per_section):
        glBindBuffer(GL_ARRAY_BUFFER, cylinder_vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        # Draw each cylinder section
        for i in range(len(self.track_points)):
            glDrawArrays(GL_TRIANGLE_STRIP, i * num_vertices_per_section, num_vertices_per_section)

        glDisableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def load_track_data(self, filename):
        # Load the entire file into a NumPy array
        data = np.loadtxt(filename, delimiter='\t', skiprows=1)

        # Process the data
        for row in data:
            pos = Vector(row[1], row[2], row[3])
            front = Vector(row[4], row[5], row[6])
            left = Vector(row[7], row[8], row[9])
            up = Vector(row[10], row[11], row[12])

            track_point = TrackPoint(pos, front, left, up)
            self.track_positions.append(pos)
            self.track_points.append(track_point)
            
        self.compute_track_normals()

    def calculate_total_length(self):
        total_length = 0
        for i in range(1, len(self.track_positions)):
            p1 = self.track_positions[i - 1]
            p2 = self.track_positions[i]
            segment_length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
            total_length += segment_length
        return total_length

    def compute_track_normals(self):
        # Assuming your track data is a series of triangles
        vertices = [tp.pos.components for tp in self.track_points]
        normals = compute_normals(vertices)  # Utilize the compute_normals function
        self.track_normals = normals  
        
    def create_mesh_between_rings(self, ring1, ring2):
        mesh_triangles = []
        num_vertices = len(ring1['inner_circle'])  # Assuming both rings have the same number of vertices

        for i in range(num_vertices):
            next_i = (i + 1) % num_vertices  # Loop back to the start for the last vertex

            # Triangle 1
            triangle1 = [ring1['inner_circle'][i], ring2['inner_circle'][i], ring2['inner_circle'][next_i]]
            mesh_triangles.append(triangle1)

            # Triangle 2
            triangle2 = [ring1['inner_circle'][i], ring2['inner_circle'][next_i], ring1['inner_circle'][next_i]]
            mesh_triangles.append(triangle2)

        return mesh_triangles

    def calculate_bisecting_planes(self):        
        for i in range(1, len(self.track_points) - 1):
            self.plane_normals.append(self.track_points[i].front)

    def generate_circles(self, num_segments=16, radius=2.5):
        angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)

        for track_point in self.track_points:
            # Compute rotation matrices for the current track point's front vector
            rotation_matrices = self.compute_rotation_matrices(track_point.front.normalized, angles)

            # Apply the rotation matrices to the 'up' vector of the track point
            rotated_vectors = np.dot(rotation_matrices, track_point.up.components)

            # Scale and translate to form the circle vertices
            circle_vertices = rotated_vectors * radius + track_point.pos.components
            track_point.circle_vertices = circle_vertices.tolist()



    def compute_rotation_matrices(self, axis, angles):
        k = np.array(axis.normalized)
        cos_theta = np.cos(angles)
        sin_theta = np.sin(angles)

        kx, ky, kz = k
        kx2, ky2, kz2 = kx * kx, ky * ky, kz * kz
        kxky, kxkz, kykz = kx * ky, kx * kz, ky * kz
        kx_sin, ky_sin, kz_sin = kx * sin_theta, ky * sin_theta, kz * sin_theta

        mat = np.zeros((len(angles), 3, 3))
        mat[:, 0, 0] = cos_theta + kx2 * (1 - cos_theta)
        mat[:, 0, 1] = kxky * (1 - cos_theta) - kz_sin
        mat[:, 0, 2] = kxkz * (1 - cos_theta) + ky_sin
        mat[:, 1, 0] = kxky * (1 - cos_theta) + kz_sin
        mat[:, 1, 1] = cos_theta + ky2 * (1 - cos_theta)
        mat[:, 1, 2] = kykz * (1 - cos_theta) - kx_sin
        mat[:, 2, 0] = kxkz * (1 - cos_theta) - ky_sin
        mat[:, 2, 1] = kykz * (1 - cos_theta) + kx_sin
        mat[:, 2, 2] = cos_theta + kz2 * (1 - cos_theta)

        return mat



    def apply_rotation_matrix(self, vec, rotation_matrix):
        return np.dot(rotation_matrix, np.array(vec))
        
    @staticmethod
    def calculate_circle_vertices(center, up, front, radius, num_segments):
        angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        circle_vertices = []
        
        for angle in angles:
            rotated_vector = Track.rotate_vector(up, front, angle)
            vertex = center + rotated_vector * radius
            circle_vertices.append(vertex.components)

        return circle_vertices
    
    @staticmethod
    def rotate_vector(vec, axis, angle):
        k = axis.normalized
        vec = Vector(vec)
        rotated_vec = vec * math.cos(angle) + Vector.Cross(k, vec) * math.sin(angle) + k * (k * vec) * (1 - math.cos(angle))
        return rotated_vec

    def compute_plane_vertices(self, size=1.0):
        # Extract positions, left, and up vectors into NumPy arrays
        positions = np.array([tp.pos for tp in self.track_points])
        left_vectors = np.array([tp.left * size / 2 for tp in self.track_points])
        up_vectors = np.array([tp.up * size / 2 for tp in self.track_points])

        # Vectorized computation of vertices
        vertices = np.empty((len(self.track_points), 4, 3))  # Shape: (num_track_points, 4 vertices, 3 coordinates)
        vertices[:, 0, :] = positions - left_vectors - up_vectors
        vertices[:, 1, :] = positions + left_vectors - up_vectors
        vertices[:, 2, :] = positions + left_vectors + up_vectors
        vertices[:, 3, :] = positions - left_vectors + up_vectors

        # Flatten the array and convert it back to a list of Vector objects
        self.plane_vertices = vertices.reshape(-1, 3).tolist()

def load_skybox_texture(faces):
    texture_id = glGenTextures(1)
    glActiveTexture(GL_TEXTURE0)    
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id)

    for i, face in enumerate(faces):
        # Load the face image
        image = Image.open(f"textures/{face}")
        img_data = np.array(image, dtype=np.uint8)


        # Specify the texture for this face
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, image.size[0], image.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    # Set texture parameters
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0)  # Unbind the texture
    return texture_id



def create_circles_vbo(track_points):
    # Aggregate all circle vertices into a single array
    all_circle_vertices = []
    for track_point in track_points:
        circle_vertices = track_point.circle_vertices
        all_circle_vertices.extend(circle_vertices)
    circle_vertex_data = np.array(all_circle_vertices, dtype=np.float32)

    # Generate and bind a VBO
    circle_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, circle_vbo)

    # Upload the aggregated circle data to the VBO
    glBufferData(GL_ARRAY_BUFFER, circle_vertex_data.nbytes, circle_vertex_data, GL_STATIC_DRAW)

    # Unbind the buffer
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    return circle_vbo

def draw_circles(circles_vbo, num_circles, num_vertices_per_circle, color=(0, 1, 0)):
    glBindBuffer(GL_ARRAY_BUFFER, circles_vbo)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

    # Set circle color
    glColor3fv(color)

    # Draw all circles in a single call
    for i in range(num_circles):
        glDrawArrays(GL_LINE_LOOP, i * num_vertices_per_circle, num_vertices_per_circle)

    glDisableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)


def draw_track(track_vbo, track_normals_vbo, num_points):
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)

    glBindBuffer(GL_ARRAY_BUFFER, track_vbo)
    glVertexPointer(3, GL_FLOAT, 0, None)

    glBindBuffer(GL_ARRAY_BUFFER, track_normals_vbo)
    glNormalPointer(GL_FLOAT, 0, None)

    glDrawArrays(GL_TRIANGLES, 0, num_points)

    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)

def create_track_vbo(track_positions):
    # Convert track_points to a suitable numpy array format
    track_point_data = np.array(track_positions, dtype=np.float32)

    # Generate and bind a VBO
    track_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, track_vbo)

    # Upload the track data to the VBO
    glBufferData(GL_ARRAY_BUFFER, track_point_data.nbytes, track_point_data, GL_STATIC_DRAW)

    # Unbind the buffer
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    return track_vbo


def calculate_bounding_box(track_positions):
    min_coords = np.min(track_positions, axis=0)
    max_coords = np.max(track_positions, axis=0)
    return min_coords, max_coords

def draw_axes(length=1.0, position=(0, 0, 0)):
    glPushMatrix()  # Save the current matrix
    glTranslate(*position)  # Move to the specified position

    # X Axis in red
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(length, 0, 0)
    glEnd()

    # Y Axis in green
    glColor3f(0.0, 1.0, 0.0)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(0, length, 0)
    glEnd()

    # Z Axis in blue
    glColor3f(0.0, 0.0, 1.0)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, length)
    glEnd()

    glPopMatrix()  # Restore the original matrix


def render_axes():
    # Render the axes with orthographic projection
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()  # Save current projection matrix
    glLoadIdentity()
    glOrtho(-1, 1, -1, 1, -1, 1)  # Set up an orthogonal projection
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()  # Save current modelview matrix
    glLoadIdentity()
    glTranslate(-0.8, 0.8, 0)  # Position the axes in the corner
    draw_axes(0.1)  # Smaller axes length for corner display
    glPopMatrix()  # Restore original modelview matrix
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()  # Restore original projection matrix
    glMatrixMode(GL_MODELVIEW)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

# Texture loading function (generalized for loading any texture)
def load_texture(texture_path):
    texture_surface = pygame.image.load(texture_path).convert_alpha()
    texture_data = pygame.image.tostring(texture_surface, "RGBA", True)
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture_surface.get_width(), texture_surface.get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return texture

def create_grass(tile_width, tile_height, divisions_width, divisions_height, center_x, center_y, grass_texture):
    vertices = []
    texture_coords = []

    # Calculate the starting point based on the center
    start_x = center_x - (divisions_width * tile_width) / 2
    start_y = center_y - (divisions_height * tile_height) / 2

    for y in range(divisions_height):
        for x in range(divisions_width):
            # Position of each tile based on the starting point
            x0 = start_x + x * tile_width
            y0 = start_y + y * tile_height
            x1 = x0 + tile_width
            y1 = y0 + tile_height

            # Append vertices for a tile (two triangles making up a rectangle)
            vertices.extend([x0, 0, y0, x1, 0, y0, x1, 0, y1, x0, 0, y1])
            
            # Texture coordinates for each vertex
            tx0 = x / divisions_width
            ty0 = y / divisions_height
            tx1 = (x + 1) / divisions_width
            ty1 = (y + 1) / divisions_height
            texture_coords.extend([tx0, ty0, tx1, ty0, tx1, ty1, tx0, ty1])

    return vertices, texture_coords, grass_texture


# Sky creation function
def create_sky():
    skybox_vertices = [
        -500,  500, 500,   500,  500,  500,   500, -500,  500,  -500, -500,  500,  # Front face
        -500, -500, -500,  -500,  500, -500,   500,  500, -500,   500, -500, -500,  # Back face
        -500,  500, -500,  -500,  500,  500,   500,  500,  500,   500,  500, -500,  # Top face
        -500, -500, -500,   500, -500, -500,   500, -500,  500,  -500, -500,  500,  # Bottom face
        500, -500, -500,   500,  500, -500,   500,  500,  500,   500, -500,  500,  # Right face
        -500, -500, -500,  -500, -500,  500,  -500,  500,  500,  -500,  500, -500   # Left face
    ]

    # Each face of the skybox should have 4 vertices, each with 2 texture coordinates (u, v)
    skybox_texture_coords = [
        0, 0, 1, 0, 1, 1, 0, 1,
        1, 0, 0, 0, 0, 1, 1, 1,
        0, 1, 0, 0, 1, 0, 1, 1,
        1, 1, 0, 1, 0, 0, 1, 0,
        1, 0, 1, 1, 0, 1, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 0
    ]


    return skybox_vertices, skybox_texture_coords

# Grass rendering function
def render_grass(vertices, texture_coords, texture):
    glColor3f(1.0, 1.0, 1.0)    
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture)

    glBegin(GL_QUADS)
    for i in range(len(vertices) // 3):
        glTexCoord2fv(texture_coords[i * 2: i * 2 + 2])
        glVertex3fv(vertices[i * 3: i * 3 + 3])
    glEnd()

    glDisable(GL_TEXTURE_2D)


def render_sky(skybox_vbo, skybox_texture_coords_vbo, texture_id, num_vertices):
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id)
    
    glBindBuffer(GL_ARRAY_BUFFER, skybox_vbo)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, skybox_texture_coords_vbo)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(1)

    glDrawArrays(GL_QUADS, 0, num_vertices)

    glDisableVertexAttribArray(0)
    glDisableVertexAttribArray(1)


def draw_planes(plane_vbo, num_planes):
    glBindBuffer(GL_ARRAY_BUFFER, plane_vbo)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    
    for i in range(num_planes):
        glDrawArrays(GL_QUADS, i * 4, 4)
    
    glDisableVertexAttribArray(0)

def setup_lighting():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)  # Enable a light source

    # Set light properties (position, color, etc.)
    glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 10.0, 10.0, 1.0])  # Positional light
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])     # Ambient light
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.5, 0.5, 0.5, 1.0])     # Diffuse light
    glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])   
    
def set_track_material():
    # Material properties for the track
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.6, 0.6, 0.6, 1.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.8, 0.8, 0.8, 1.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 50.0)   
    
def compute_normals(vertices):
    normals = []
    num_vertices = len(vertices)
    
    for i in range(num_vertices - 2):
        v1 = np.array(vertices[i])
        v2 = np.array(vertices[i + 1])
        v3 = np.array(vertices[i + 2])

        # Determine the winding order
        if i % 2 == 0:
            u = v3 - v1
            v = v2 - v1
        else:
            u = v2 - v1
            v = v3 - v1

        normal = np.cross(u, v)
        normal = normal / np.linalg.norm(normal)  # Normalize the normal

        # Add the normal for each vertex in the triangle
        normals.extend([normal.tolist()] * 3)

    # For a closed loop, the last two triangles' normals need special handling
    if len(vertices) % 2 == 0:  # Adjust if your vertex count is even
        normals[-6:-3] = normals[-3:] = normals[-6:-3]  # Copy the second last normal to the last

    return normals
   

def create_normals_vbo(normals):
    normals_data = np.array(normals, dtype=np.float32)
    normals_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, normals_vbo)
    glBufferData(GL_ARRAY_BUFFER, normals_data.nbytes, normals_data, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return normals_vbo

def compile_shader(shader_code, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, shader_code)
    glCompileShader(shader)

    # Check for compilation errors
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        print(f'Error compiling shader: {error}')
        glDeleteShader(shader)
        return None

    return shader

def create_shader_program_from_files(vertex_shader_file, fragment_shader_file):
    vertex_shader_code = load_shader_source(vertex_shader_file)
    fragment_shader_code = load_shader_source(fragment_shader_file)

    vertex_shader = compile_shader(vertex_shader_code, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_shader_code, GL_FRAGMENT_SHADER)

    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    # Check for linking errors
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        print(f'Error linking program: {error}')
        glDeleteProgram(program)
        return None

    glDetachShader(program, vertex_shader)
    glDetachShader(program, fragment_shader)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return program

def load_shader_source(shader_file):
    with open(shader_file, 'r') as file:
        shader_source = file.read()
    return shader_source


def update_light_position(camera_position):
    # Assuming camera_position is a Vector with x, y, and z attributes
    light_position = [camera_position.x, camera_position.y, camera_position.z, 1.0]  # The last element is 1.0 for a positional light
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)



def show_tk_splash_screen(splash_image_path, close_event):
    root = tk.Tk()
    root.overrideredirect(True)  # Remove window decorations
    root.attributes("-topmost", True)
    
    # Load the splash image using PIL
    img = Image.open(splash_image_path)
    tk_image = ImageTk.PhotoImage(img)
    
    tk.Label(root, image=tk_image).pack()

    # Center the splash screen
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - img.width) // 2
    y = (screen_height - img.height) // 2
    root.geometry(f'+{x}+{y}')

    # Function to check the event and close the window
    def check_close():
        if close_event.is_set():
            root.destroy()
        else:
            root.after(100, check_close)  # Check again after 100ms

    root.after(100, check_close)  # Start checking for the close event
    root.mainloop()

def initialize_track():
    pygame.init()
    display = (800, 600)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    grass_texture = load_texture("textures/grass.png")
    
    track = Track()
    track.load_track_data('Two Arrows.csv')
    track_vbo = create_track_vbo(track.track_positions)
    track_normals_vbo = create_normals_vbo(track.track_normals)       
    track.calculate_bisecting_planes()

    track.compute_plane_vertices(size=1)
    cylinder_vbo = track.generate_and_bind_cylinder_vbo(radius=0.5, num_segments=16)
    
    plane_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, plane_vbo)

    plane_vertex_data = np.array(track.plane_vertices, dtype=np.float32)
    glBufferData(GL_ARRAY_BUFFER, plane_vertex_data.nbytes, plane_vertex_data, GL_STATIC_DRAW)

    # Add other initializations like setup_lighting, set_track_material, etc.
    min_coords, max_coords = calculate_bounding_box(track.track_positions)
    bbox_center = (min_coords + max_coords) / 2
    bbox_size = max_coords - min_coords

    grass_vertices, grass_texture_coords, _ = create_grass(100, 100, 10, 10, bbox_center[0], bbox_center[1], grass_texture)
    skybox_vertices, skybox_texture_coords = create_sky()
    faces = ['posx.jpg', 'negx.jpg', 'posy.jpg', 'negy.jpg', 'posz.jpg', 'negz.jpg']
    skybox_texture_id = load_skybox_texture(faces)

    # Generate VBOs for skybox
    # Convert to numpy arrays
    vertices_array = np.array(skybox_vertices, dtype=np.float32)
    texture_coords_array = np.array(skybox_texture_coords, dtype=np.float32)

    # Generate VBOs
    skybox_vbo = glGenBuffers(1)
    skybox_texture_coords_vbo = glGenBuffers(1)

    # Bind and fill the vertex VBO
    glBindBuffer(GL_ARRAY_BUFFER, skybox_vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices_array.nbytes, vertices_array, GL_STATIC_DRAW)

    # Bind and fill the texture coordinate VBO
    glBindBuffer(GL_ARRAY_BUFFER, skybox_texture_coords_vbo)
    glBufferData(GL_ARRAY_BUFFER, texture_coords_array.nbytes, texture_coords_array, GL_STATIC_DRAW)

    # Unbind the buffer (optional, for cleanup)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    shader_program = create_shader_program_from_files("VertexShader.glsl", "FragmentShader.glsl")

    # Package relevant variables in a dictionary or a custom class to return them
    return {
        "screen": screen,
        "track_vbo": track_vbo,
        "track_normals_vbo": track_normals_vbo,
        "cylinder_vbo": cylinder_vbo,
        "plane_vbo": plane_vbo,
        "grass_vertices": grass_vertices,
        "grass_texture_coords": grass_texture_coords,
        "grass_texture": grass_texture,
        "skybox_vbo": skybox_vbo,
        "skybox_texture_coords_vbo": skybox_texture_coords_vbo,
        "skybox_texture_id": skybox_texture_id,
        "shader_program": shader_program,
        "bbox_size": bbox_size,
        "camera_position": Vector(-100,10,0),
        "skybox_vertices": skybox_vertices,
        "skybox_texture_coords": skybox_texture_coords,
        "display": display,
        "track": track
    }



def main():   
    close_event = threading.Event()

    # Start the splash screen
    image_number = random.randint(1, 4)     
    splash_thread = threading.Thread(target=show_tk_splash_screen, args=(f'textures/splash{image_number}.jpg', close_event))    
    splash_thread.start()

    # Perform the track initialization
    init_data = initialize_track()

    # Signal the splash screen to close
    close_event.set()
    splash_thread.join()    
    
    # Close the splash screen window and start the main application window
    # pygame.quit()
    display = (800, 600)
    # Extract variables for ease of use
    screen = init_data["screen"]
    track_vbo = init_data["track_vbo"]
    track_normals_vbo = init_data["track_normals_vbo"]
    cylinder_vbo = init_data["cylinder_vbo"]
    plane_vbo = init_data["plane_vbo"]
    grass_vertices = init_data["grass_vertices"]
    grass_texture_coords = init_data["grass_texture_coords"]
    grass_texture = init_data["grass_texture"]
    skybox_vbo = init_data["skybox_vbo"]
    skybox_texture_coords_vbo = init_data["skybox_texture_coords_vbo"]
    skybox_texture_id = init_data["skybox_texture_id"]
    bbox_size = init_data["bbox_size"]
    camera_position = init_data["camera_position"]
    skybox_vertices = init_data["skybox_vertices"]
    skybox_texture_coords = init_data["skybox_texture_coords"]
    display = init_data["display"]
    track = init_data["track"]
    

    camera_position: Vector = Vector(-100,10,0)

    x_move, y_move, z_move = 0, 0, 0
    cumulative_x_rot, cumulative_y_rot = 0, 0

    forward_pressed = False 
    backward_pressed = False
    left_pressed = False
    right_pressed = False
    up_pressed = False
    down_pressed = False
    shift_pressed = False 
    right = Vector([-1, 0, 0])
    up = Vector([0, 1, 0])
    yaw = 180 * math.pi / 180 
    pitch = 0

    front = Vector([0, 0, 1])
    clock = pygame.time.Clock()

    # Extract variables for ease of use
    screen = init_data["screen"]
    track_vbo = init_data["track_vbo"]
    track_normals_vbo = init_data["track_normals_vbo"]
    cylinder_vbo = init_data["cylinder_vbo"]
    plane_vbo = init_data["plane_vbo"]
    grass_vertices = init_data["grass_vertices"]
    grass_texture_coords = init_data["grass_texture_coords"]
    grass_texture = init_data["grass_texture"]
    skybox_vbo = init_data["skybox_vbo"]
    skybox_texture_coords_vbo = init_data["skybox_texture_coords_vbo"]
    skybox_texture_id = init_data["skybox_texture_id"]
    shader_program = init_data["shader_program"]
    bbox_size = init_data["bbox_size"]
    camera_position = init_data["camera_position"]

    # Main loop
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                # check if shift is pressed
                if event.mod & pygame.KMOD_SHIFT:
                    shift_pressed = True
                if event.key == pygame.K_w:
                    forward_pressed = True
                elif event.key == pygame.K_s:
                    backward_pressed = True
                elif event.key == pygame.K_a:
                    left_pressed = True
                elif event.key == pygame.K_d:
                    right_pressed = True
                elif event.key == pygame.K_e:
                    up_pressed = True
                elif event.key == pygame.K_q:
                    down_pressed = True                    
                elif event.key == pygame.K_ESCAPE:
                    running = False

            if event.type == pygame.KEYUP:
                shift_pressed = False
                if event.key == pygame.K_w:
                    forward_pressed = False
                elif event.key == pygame.K_s:
                    backward_pressed = False
                elif event.key == pygame.K_a:
                    left_pressed = False
                elif event.key == pygame.K_d:
                    right_pressed = False
                elif event.key == pygame.K_e:
                    up_pressed = False
                elif event.key == pygame.K_q:
                    down_pressed = False                      

            if event.type == pygame.MOUSEMOTION:
                yaw += event.rel[0] * 0.001
                pitch -= event.rel[1] * 0.001                


        # Calculate movement vectors based on camera's local orientation
        forward_movement: Vector = front / 10
        right_movement: Vector = right / 10
        up_movement:Vector = up / 10
        if shift_pressed:
            forward_movement *= 10
            right_movement *= 10
            up_movement *= 10

        # Apply movement based on key presses
        if forward_pressed:
            camera_position += forward_movement
        elif backward_pressed:
            camera_position -= forward_movement

        if left_pressed:
            camera_position -= right_movement
        elif right_pressed:
            camera_position += right_movement

        if up_pressed:
            camera_position += up_movement
        elif down_pressed:
            camera_position -= up_movement            

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(shader_program)
        # Set up perspective projection for the main scene
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (display[0] / display[1]), 0.1, 5000.0)

        front: Vector = Vector(math.cos(yaw) * math.cos(pitch), math.sin(pitch), math.sin(yaw) * math.cos(pitch))

        front = normalize(front)

        right = Vector.Cross(front, up)
        right = normalize(right) 

        up = normalize(up)

        # gluLookAt 
        gluLookAt(camera_position.x, camera_position.y, camera_position.z,
                camera_position.x + front.x, 
                camera_position.y + front.y, 
                camera_position.z + front.z,
                up.z, up.y, up.z)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glRotatef(cumulative_x_rot, 1, 0, 0)
        glRotatef(cumulative_y_rot, 0, 1, 0)
        glTranslate(-camera_position.x, -camera_position.y, -camera_position.z)
        update_light_position(camera_position)
        
        glEnable(GL_DEPTH_TEST)
        render_sky(skybox_vbo, skybox_texture_coords_vbo, skybox_texture_id, len(skybox_vertices))
        render_grass(grass_vertices, grass_texture_coords, grass_texture)

        glColor3f(0.0, 1.0, 0.0)
        glLineWidth(2.0)
        draw_track(track_vbo, track_normals_vbo, len(track.track_positions))
        glColor3f(1.0, 0.0, 0.0)  # Red color for cylinders
        track.draw_cylinders(cylinder_vbo, num_vertices_per_section=32)  # 32 vertices per section (16 top + 16 bottom)
        
        glColor3f(1.0, 0.0, 0.0)  # Set plane color
        # draw_planes(plane_vbo, len(track.plane_normals))
        # for track_point in track.track_points:
        #     circle_vbo = create_circle_vbo(track_point.circle_vertices)
        #     draw_circle(circle_vbo, 16, color=(0, 0, 1))
        # circle_vbo = create_circles_vbo(track.track_points)        
        # draw_circles(circle_vbo, len(track.track_points), 16, color=(0.5, 0.5, 1))
        # Draw the axes in a fixed position relative to the camera
        glBindBuffer(GL_ARRAY_BUFFER, cylinder_vbo)
        glEnableVertexAttribArray(0)  # Assuming location 0 for vertex positions
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        # Draw the cylinders
        glDrawArrays(GL_TRIANGLE_STRIP, 0, len(track.track_points) * len(track.track_points) * 2)

        glDisableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)        
        axes_position = (-3, -3, -5)  # Adjust this as needed for visibility
        draw_axes(length=2.0, position=axes_position)

        pygame.display.flip()
        # clock.tick(60)

if __name__ == "__main__":
    main()