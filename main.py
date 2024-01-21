import csv
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

def load_track_data(filename):
    track_points = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)  # Skip header
        for row in reader:
            point = (float(row[1]), float(row[2]), float(row[3]))
            track_points.append(point)
    return track_points

def calculate_bounding_box(track_points):
    min_coords = np.min(track_points, axis=0)
    max_coords = np.max(track_points, axis=0)
    return min_coords, max_coords

def draw_track(track_points):
    glBegin(GL_LINE_STRIP)
    for point in track_points:
        glVertex3fv(point)
    glEnd()

def draw_axes(length=1.0):
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

def main():
    pygame.init()
    display = (800, 600)
    screen = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    track_data = load_track_data('Two Arrows.csv')  # Replace with your CSV file path
    min_coords, max_coords = calculate_bounding_box(track_data)
    bbox_center = (min_coords + max_coords) / 2
    bbox_size = max_coords - min_coords

    camera_distance = max(bbox_size) * 2
    camera_position = bbox_center + np.array([0, 0, camera_distance])

    gluPerspective(45, (display[0]/display[1]), 0.1, 500.0)
    gluLookAt(camera_position[0], camera_position[1], camera_position[2], 
              bbox_center[0], bbox_center[1], bbox_center[2], 
              0, 1, 0)

    x_move, y_move, z_move = 0, 0, 0
    x_rot, y_rot = 0, 0

    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    z_move = -0.1
                elif event.key == pygame.K_s:
                    z_move = 0.1
                if event.key == pygame.K_a:
                    x_move = -0.1
                elif event.key == pygame.K_d:
                    x_move = 0.1

            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_w, pygame.K_s]:
                    z_move = 0
                if event.key in [pygame.K_a, pygame.K_d]:
                    x_move = 0

            if event.type == pygame.MOUSEMOTION:
                x_rot = event.rel[1]
                y_rot = event.rel[0]

        glRotatef(x_rot, 1, 0, 0)
        glRotatef(y_rot, 0, 1, 0)
        glTranslatef(x_move, y_move, z_move)

        # Render main scene (roller coaster)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glColor3f(0.0, 1.0, 0.0)
        glLineWidth(2.0)
        draw_track(track_data)

        # Save current matrices
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, display[0], 0, display[1])

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Set viewport for the axes in the corner
        glViewport(0, 0, display[0] // 4, display[1] // 4)

        # Draw axes
        draw_axes(0.8)

        # Restore original matrices and viewport
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glViewport(0, 0, display[0], display[1])

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()