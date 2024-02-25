import freetype
import re
import tkinter as tk
from PIL import Image, ImageTk
import threading
import random
import time
import math
import csv
import numpy as np
from PIL import Image
from madcad import *
import freetype
from pygame import ver
from tqdm import tqdm
            
# Assuming these are the radii for each track, defined somewhere in your code
TRACK_RADIUS = 0.1
POST_RADIUS = 0.075
SUPPORT_RADIUS = 0.3
SMALL_SUPPORT_LENGTH = 1

O = vec3(0,0,0)
X = vec3(1,0,0)
Y = vec3(0,1,0)
Z = vec3(0,0,1)

def select_left_track(track_point):
    return track_point.left_track_pos(), TRACK_RADIUS

def select_right_track(track_point):
    return track_point.right_track_pos(), TRACK_RADIUS

def select_center_track(track_point):
    return track_point.center_track_pos(), TRACK_RADIUS

class Plane:
    def __init__(self, point, normal):
        self.point = point  # A point on the plane
        self.normal = normal  # Normal vector to the plane

    @classmethod
    def from_points(cls, p1, p2, p3):
        normal = cls.calculate_normal(p1, p2, p3)
        return cls(p1, normal)
    
    @classmethod
    def calculate_normal(cls, p1, p2, p3):
        """Calculate the normal vector of the plane defined by points p1, p2, p3."""
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)  # Normalize the normal vector
        return normal

# Each TrackPoint is like that in the file:
# "No."	"PosX"	"PosY"	"PosZ"	"FrontX"	"FrontY"	"FrontZ"	"LeftX"	"LeftY"	"LeftZ"	"UpX"	"UpY"	"UpZ"
# 1	-290.606498	10.777089	29.620222	-0.997019	-0.043619	0.063648	0.063708	-0.000000	0.997969	-0.043531	0.999048	0.002779
# 2	-292.603033	10.700273	29.697521	-0.999306	-0.036561	-0.007144	-0.007087	-0.001686	0.999973	-0.036572	0.999330	0.001425
# 3	-294.595254	10.625287	29.557128	-0.989779	-0.039035	-0.137166	-0.137124	-0.003764	0.990547	-0.039182	0.999231	-0.001627        
class TrackPoint:     
    def __init__(self, pos, front, left, up, track_width, center_track_lowering):
        self.planes = []
        self.pos = vec3(pos)
        self.front = vec3(front)
        self.left = vec3(left)
        self.up = vec3(up)
        self.down = vec3(-up.x, -up.y, -up.z)
        self.right = vec3(-left.x, -left.y, -left.z)
        self.back = vec3(-front.x, -front.y, -front.z)
        self.track_width = track_width
        self.center_track_lowering = center_track_lowering
        self.circle_vertices = []        

    def is_inverted(self):
        return self.up.z < 0

    def select_track_position(self, position='center'):
        if position == 'left':
            return self.left_track_pos()
        elif position == 'right':
            return self.right_track_pos()
        elif position == 'center':
            return self.center_track_pos()
        else:
            raise ValueError("Invalid track position selected")
    def left_track_pos(self):
        return self.pos - self.left * self.track_width / 2

    def right_track_pos(self):
        return self.pos + self.left * self.track_width / 2

    def center_track_pos(self):
        return self.pos - self.up * self.center_track_lowering
    
    def rotation_matrix(self):
        front_normalized = normalize(self.front)
        up_normalized = normalize(self.up)

        # Calculate the third vec3 using cross product to form an orthogonal basis
        right = vec3(-self.left.x, -self.left.y, -self.left.z)

        # Rotation matrix with 'right', 'up_normalized', and 'front_normalized' as its columns
        rot_matrix = mat4(
            right.x, right.y, right.z, 0, 
            up_normalized.x, up_normalized.y, up_normalized.z, 0,
            front_normalized.x, front_normalized.y, front_normalized.z, 0,
            0, 0, 0, 1
        )
        return rot_matrix
    
    def lean_direction(self):
        world_up = vec3(0, 0, 1)
        
        # Check if front is parallel or almost parallel to world_up
        if abs(dot(self.front, world_up)) >= 0.999:
            # front is parallel to world_up, choose a default horizontal direction
            horizontal_perpendicular = vec3(1, 0, 0)
        else:
            # Compute the perpendicular horizontal vector
            horizontal_perpendicular = normalize(cross(self.front, world_up))
        
        # Now, calculate the dot product between the left vector and this horizontal perpendicular
        dot_product = dot(self.left, horizontal_perpendicular)
        if dot_product > 0:
            return 1
        elif dot_product < 0:
            return -1
        else:
            return 0
        
    def ignore_based_on_angle(self, threshold_angle_degrees):
        world_up = vec3(0, 0, 1)
        # Normalize vectors to unit vectors
        world_up_normalized = normalize(world_up)
        track_down_normalized = normalize(self.down)
        
        # Calculate dot product
        dot_product = dot(world_up_normalized, track_down_normalized)
        
        # Calculate the angle in radians between the vectors
        angle_radians = np.arccos(dot_product)
        
        # Convert threshold from degrees to radians
        threshold_angle_radians = np.radians(threshold_angle_degrees)
        
        # Check if the angle is mainer than the threshold
        if angle_radians < threshold_angle_radians:
            return True  # Ignore this trackPos
        else:
            return False  # Don't ignore this trackPos        

    
class Track:
    def __init__(self):
        self.track_points = []
        self.track_positions = []
        self.post_number = 1 
        self.max_angle = 0
        self.avg_angle = 0
        self.num_angle = 0
        self.total_angle = 0
        self.total_length = 0

    def check_line_intersection(self, p1, p2):
        for track_point in self.track_points:
            for plane in track_point.planes:
                intersection = self.line_plane_intersection(plane, p1, p2 - p1)
                if intersection is not None:
                    return intersection

    def compute_planes(self):
        """Compute and store planes for each TrackPoint based on the provided specifications."""
        for i in range(len(self.track_points) - 1):
            plane1 = Plane.from_points(self.track_points[i].left_track_pos(), self.track_points[i].center_track_pos(), self.track_points[i+1].left_track_pos())
            plane2 = Plane.from_points(self.track_points[i].center_track_pos(), self.track_points[i].right_track_pos(), self.track_points[i+1].right_track_pos())
            plane3 = Plane.from_points(self.track_points[i].right_track_pos(), self.track_points[i].left_track_pos(), self.track_points[i+1].right_track_pos())            

            self.track_points[i].planes = [plane1, plane2, plane3]

    def curvature(self, track_point1, track_point2, track_point3):
        # Extract positions as NumPy arrays for easier calculations
        p1 = np.array([track_point1.pos.x, track_point1.pos.y, track_point1.pos.z])
        p2 = np.array([track_point2.pos.x, track_point2.pos.y, track_point2.pos.z])
        p3 = np.array([track_point3.pos.x, track_point3.pos.y, track_point3.pos.z])

        # Use the curvature calculation formula
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        try:
            curv = 4 * area / (a * b * c)
        except ZeroDivisionError:
            curv = 0
        return curv


    def line_plane_intersection(self, plane, position, direction):
        # Assuming ground plane has a normal of (0, 0, 1) and goes through (0, 0, 0)
        plane.normal = vec3(0, 0, 1)
        plane.point = vec3(0, 0, 0)
        
        # Normalize the direction to ensure proper scaling
        direction_normalized = normalize(direction)
        
        # Calculate t using the correct formula
        numerator = dot(plane.point - position,plane.normal)
        denominator = dot(direction_normalized,plane.normal)
        
        # To avoid division by zero, check if the denominator is non-zero
        if denominator == 0:
            # The direction is parallel to the ground plane; no intersection
            return None, None
        
        t = numerator / denominator
        
        # Calculate intersection point using the correctly scaled direction
        intersection_point = position + direction_normalized * t
        
        # Calculate the length of the post as the distance from the post_base to the intersection_point
        support_length = len(intersection_point - position)
        
        return support_length, intersection_point

    def calculate_intersection_with_ground(self, post_base, direction):
        # Assuming ground plane has a normal of (0, 0, 1) and goes through (0, 0, 0)
        ground_normal = vec3(0, 0, 1)
        ground_point = vec3(0, 0, 0)
        ground_plane = Plane(ground_point, ground_normal)
        
        return self.line_plane_intersection(ground_plane, post_base, direction)


    def generate_supports(self, base_interval, min_interval):
        accumulated_length = 0.0
        support_mesh = Mesh()  # Assuming Mesh() is defined elsewhere


        # def calculate_turn_angle(curvature, arc_length):
        #     # Calculate the angle in radians
        #     angle_radians = curvature * arc_length
            
        #     # Convert the angle to degrees
        #     angle_degrees = math.degrees(angle_radians)
            
        #     return angle_degrees

        # def calculate_circumcircle_radius(p1, p2, p3):
        #     # Convert points to vec3 if not already
        #     p1, p2, p3 = vec3(p1), vec3(p2), vec3(p3)
            
        #     # Calculate the lengths of the triangle's sides using len() for magnitude
        #     a = len(p2 - p3)
        #     b = len(p1 - p3)
        #     c = len(p1 - p2)
            
        #     # Semi-perimeter
        #     s = (a + b + c) / 2
            
        #     # Triangle area using Heron's formula and cross product magnitude for area of parallelogram
        #     area = len(cross(p2-p1, p3-p1)) / 2
            
        #     # Circumcircle radius using the triangle area and side lengths
        #     R = (a * b * c) / (4 * area)
        #     return R

        # def calculate_arc_length(p1, p2, p3):
        #     # Ensure inputs are vec3
        #     p1, p2, p3 = vec3(p1), vec3(p2), vec3(p3)
            
        #     # Calculate circumcircle radius
        #     R = calculate_circumcircle_radius(p1, p2, p3)
            
        #     # Use cross and dot to find the angle between (p2 - p1) and (p3 - p1)
        #     v1 = normalize(p2 - p1)
        #     v2 = normalize(p3 - p1)
        #     cross_prod = cross(v1, v2)
        #     dot_prod = dot(v1, v2)
            
        #     # Angle in radians
        #     theta = np.arctan2(len(cross_prod), dot_prod)
            
        #     # Arc length
        #     L = R * abs(theta)
        #     return L

        # def calculate_adjusted_interval(base_interval, angle_turned, min_interval):
        #     if isnan(angle_turned):
        #         return base_interval
        #     if self.max_angle < angle_turned:
        #         self.max_angle = angle_turned
        #     self.total_angle += angle_turned
        #     self.num_angle += 1            
        #     self.avg_angle = self.total_angle / self.num_angle
            
            
        #     # Scale factor for adjustment, you might want to fine-tune this value
        #     scale_factor = 3  # Example scaling factor
            
        #     # Calculate the adjustment ratio based on the angle_turned
        #     # This example assumes a linear relationship; you might explore non-linear options for different behaviors
        #     adjustment_ratio = 1 - (angle_turned / 180.0) * scale_factor  # Cap at 180 degrees for simplicity
            
        #     # Calculate the adjusted interval
        #     adjusted_interval = base_interval * adjustment_ratio
            
        #     # Clamp the adjusted interval within the specified bounds
        #     adjusted_interval = max(adjusted_interval, min_interval)
            
        #     return adjusted_interval

        # Using tqdm for progress bar
        accumulated_length = 0
        for i in tqdm(range(1, len(self.track_points) - 1), desc='Generating Supports'):
            if(i > 0):
                accumulated_length += len(self.track_points[i].pos - self.track_points[i - 1].pos)
                
            if accumulated_length >= base_interval:
                if not self.track_points[i].ignore_based_on_angle(90):
                    self.place_support(self.track_points[i], support_mesh)
                    accumulated_length = 0
        return support_mesh


    def rotate_vector_around_axis(self, v, k, theta):
        # v is the vector to be rotated
        # k is the rotation axis, assumed to be a normalized vector
        # theta is the angle in radians
        
        # Rodrigues' rotation formula
        term1 = v * np.cos(theta)
        term2 = cross(k, v) * np.sin(theta)
        term3 = k * dot(k, v) * (1 - np.cos(theta))
        
        return term1 + term2 + term3

    def create_disc_web(self, outer_radius, inner_radius, num_segments=16):
        angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
        
        # Generate points for the outer and inner circles
        outer_points = [vec3(outer_radius * np.cos(angle), outer_radius * np.sin(angle), 0) for angle in angles]
        inner_points = [vec3(inner_radius * np.cos(angle), inner_radius * np.sin(angle), 0) for angle in angles]
        
        # Combine points to form the disc's points list
        points = outer_points + inner_points
        
        # Generate edges for the outer and inner circles
        outer_edges = [(i, (i+1) % num_segments) for i in range(num_segments)]
        inner_edges = [(i + num_segments, ((i+1) % num_segments) + num_segments) for i in range(num_segments)]
        
        # Combine edges to form the disc's edges list
        edges = outer_edges + inner_edges
        
        # Create the Web object
        disc_web = Web(points, edges)
        
        return disc_web

    def create_tube(self, start_point, end_point, outer_radius, thickness, num_segments=16):
        inner_radius = outer_radius - thickness
        
        # Create the disc web representing the tube's cross-section
        disc_web = self.create_disc_web(outer_radius, inner_radius, num_segments)
        
        # Now, create the path as a list of transformation matrices from start to end point
        path_transformations = [
            transform(start_point),  # Transformation at the start
            transform(end_point)     # Transformation at the end
        ]
        direction = end_point - start_point
        # Perform the extrusion with the disc web
        tube_mesh = extrusion(direction, disc_web)

        return tube_mesh

    def place_support(self, track_point, support_mesh):
        world_up = vec3(0, 0, 1)
        world_down = vec3(0, 0, -1)
        angle_degrees = 30
        angle_radians = np.radians(angle_degrees)
        

        # Determine the horizontal_perpendicular vector as before
        if abs(dot(track_point.front, world_up)) >= 0.999:
            horizontal_perpendicular = vec3(1, 0, 0)
        else:
            horizontal_perpendicular = normalize(cross(track_point.front, world_up))

        # Calculate the rotation axis
        rotation_axis = normalize(cross(horizontal_perpendicular, world_up))

        # Rotate track_point.front around the rotation axis by angle_radians
        direction_vector = self.rotate_vector_around_axis(world_down, rotation_axis, angle_radians)
        up_dot = dot(track_point.down, world_up)
        
        if up_dot > 0:        
            pos_offset = track_point.down * SMALL_SUPPORT_LENGTH
                    
            center_pos = track_point.center_track_pos()
            center_pos_offset = center_pos + pos_offset
            # Calculate and place vertical support
            _, vertical_intersection_point = self.calculate_intersection_with_ground(center_pos_offset, vec3(0, 0, -1))
            intersect = self.check_line_intersection(center_pos_offset, vertical_intersection_point)
            if(intersect is None):
                return
            vertical_top_point = center_pos_offset + vec3(0, 0, 0.5)
            vertical_support_mesh = cylinder(bottom=vertical_intersection_point, top=vertical_top_point, radius=SUPPORT_RADIUS)
            
            # Calculate the main support from the track to the vertical support
            main_support_end = center_pos
            main_support_start = center_pos_offset
            main_direction = normalize(main_support_start - main_support_end)
            main_support_mesh = cylinder(bottom=main_support_end, top=main_support_start, radius=SUPPORT_RADIUS)
            _, main_intersection_point = self.calculate_intersection_with_ground(main_support_start, direction_vector)
            if main_intersection_point is None:
                return

            # Calculate directions to check the angle between vertical and main support
            vertical_direction = normalize(vertical_top_point - vertical_intersection_point)


            # Check the angle between the vertical post and the main post
            dot_product = dot(vertical_direction, main_direction)
            # The supports are nearly collinear, so create a combined cylinder and try to union with the angled support
            if abs(dot_product) > 0.99:
                combined_support_mesh = cylinder(bottom=vertical_intersection_point, top=center_pos, radius=SUPPORT_RADIUS)
                support_mesh += combined_support_mesh
            else:
                try:      
                    union_mesh = union(vertical_support_mesh, main_support_mesh)  # Note: Using the updated support_mesh
                    support_mesh += union_mesh  # Update support_mesh with the new union if successful
                except Exception as e:                
                    print(f"\nError during union with main support: {e}\n")
                    support_mesh += main_support_mesh
                    support_mesh += vertical_support_mesh
        else:
            # Track's down is not up, add only a long vertical support from the track to the ground
            center_pos = track_point.center_track_pos()
            _, vertical_intersection_point = self.calculate_intersection_with_ground(center_pos, world_down)
            vertical_top_point = center_pos
            vertical_support_mesh = self.create_tube(vertical_intersection_point, vertical_top_point,0.4, SUPPORT_RADIUS)
            support_mesh += vertical_support_mesh            

        def transform_mesh(mesh, matrix):
            transformed_points = [matrix * p for p in mesh.points]
            mesh.points = transformed_points

        # Function to add text to a given position with specific transformations
        def add_text_at_post(base_position, str, rotation_matrix):
            text_mesh = text.text(str,
                                font='NotoSans-Regular',
                                align=('left', 0),
                                fill=True)
            transform_matrix = transform(base_position) @ rotation_matrix             
            transform_mesh(text_mesh, transform_matrix)
            return text_mesh                    
        
        
        post_number = self.post_number
        self.post_number += 1  # Increment for next use
        
        # Prepare the text to be placed
        text_to_place = str(post_number)
        
        # Calculate the base position for the text (at the bottom of each post)
        if vertical_intersection_point:
            text_base_position: dvec3 = vertical_intersection_point - vec3(0, 0, 0.5)
        
        # Rotation to stand the text vertically
        vertical_rotation_matrix = rotatearound(np.pi / 2,vec3(0, 0, 0), vec3(1, 0, 0))
        
        # Rotation to align with the track's orientation
        alignment_rotation_matrix = rotatearound(np.arctan2(track_point.front.y, track_point.front.x), vec3(0, 0, 0), vec3(0, 1, 0))
        
        # Combine transformations for text orientation
        transformation_matrix = vertical_rotation_matrix @ alignment_rotation_matrix
        
        # Add the transformed text to the support_mesh
        support_mesh += add_text_at_post(text_base_position, text_to_place, transformation_matrix)
        
        return support_mesh
        
    def generate_cylinders_for_track(self, track_selector, num_segments=16, track_name='', thickness=0.4):
        angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=True)
        sin_angles = np.sin(angles)
        cos_angles = np.cos(angles)

        outer_radius = TRACK_RADIUS
        inner_radius = TRACK_RADIUS - thickness

        sections = []
        transformations = []

        # Create circle sections for the outer and possibly inner cylinder base, if needed
        outer_circle_points = [(outer_radius * cos_angle, outer_radius * sin_angle, 0) for cos_angle, sin_angle in zip(cos_angles, sin_angles)]
        inner_circle_points = [(inner_radius * cos_angle, inner_radius * sin_angle, 0) for cos_angle, sin_angle in zip(cos_angles, sin_angles)]
        # Optional: Define inner_circle_points similarly if you're modeling the tube with an inner cavity

        outer_circle = [vec3(vec) for vec in outer_circle_points]
        inner_circle = [vec3(vec) for vec in outer_circle_points]

        for i in tqdm(range(len(self.track_points) - 1), desc=f'Generating track {track_name}'):
            tp1 = self.track_points[i]
            tp2 = self.track_points[i + 1]

            # Get position and radius for each track point
            pos1, radius1 = track_selector(tp1)
            pos2, radius2 = track_selector(tp2)

            # Add the base circle as the section for both tp1 and tp2
            sections.append(outer_circle) 
            sections.append(inner_circle) 

            # Compute transformation matrices for tp1 and tp2 and add to transformations
            transformation1 = transform(pos1) @ tp1.rotation_matrix()
            transformation2 = transform(pos2) @ tp2.rotation_matrix()
            transformations.extend([transformation1, transformation2])

        # Create the mesh by extruding and transforming the base circle along the track
        track_mesh = extrans(section=outer_circle, transformations=transformations)
        # Repeat the process for inner_circle if you have defined it and wish to subtract it from the outer to model an actual tube

        return track_mesh


    def load_track_data(self, filename):
        # Load the entire file into a NumPy array
        data = np.loadtxt(filename, delimiter='\t', skiprows=1)
        # Process the data
        for row in data:
            # Switch y and z components for each vec3
            pos = vec3(row[1], row[3], row[2])
            front = vec3(row[4], row[6], row[5])
            left = vec3(row[7], row[9], row[8])
            up = vec3(row[10], row[12], row[11])

            track_point = TrackPoint(pos, front, left, up, 1, 0.86) # 0.86 is the height of a triangle with all sides equal to 1
            self.track_positions.append(pos)
            self.track_points.append(track_point)

                

def main():   
    print("Loading Track...")
    track = Track()
    track.load_track_data('Two Arrows_low.csv')
    track.compute_planes()
    radius=0.6
    num_segments=16    
    right_track = track.generate_cylinders_for_track(select_right_track, track_name='right')
    center_track = track.generate_cylinders_for_track(select_center_track, track_name='center')
    left_track = track.generate_cylinders_for_track(select_left_track, track_name='left')
    support_mesh = track.generate_supports(20, 1)
    # Combine all track meshes into a single list for rendering
    # all_tracks = [support_mesh, left_track]
    all_tracks = [left_track,right_track,center_track, support_mesh]

    # Render the tracks
    show(all_tracks)

if __name__ == "__main__":
    main()