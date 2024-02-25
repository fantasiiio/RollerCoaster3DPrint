import FreeCAD
import Part
import csv
from FreeCAD import Base

class TrackPoint:
    def __init__(self, pos, left):
        self.pos = Base.Vector(pos)
        self.left = Base.Vector(left)

    def left_track_pos(self):
        return self.pos - self.left * 0.5  # Adjust track width as needed

    def right_track_pos(self):
        return self.pos + self.left * 0.5

def read_track_data_from_csv(filename):
    track_points = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip header row
        for row in reader:
            pos = (float(row[1]), float(row[2]), float(row[3]))  # Position vectors
            left = (float(row[7]), float(row[8]), float(row[9]))  # Left vectors
            track_points.append(TrackPoint(pos, left))
    return track_points

def create_spline_from_track_points(track_points, track_selector):
    points = [Base.Vector(track_selector(tp)) for tp in track_points]
    spline = Part.BSplineCurve()
    spline.interpolate(points)
    return spline.toShape()

def export_splines_to_step(left_spline, right_spline, filename):
    compound = Part.Compound([left_spline, right_spline])
    compound.exportStep(filename)

# Example usage
track_points = read_track_data_from_csv('Two Arrows.csv')
left_spline = create_spline_from_track_points(track_points, TrackPoint.left_track_pos)
right_spline = create_spline_from_track_points(track_points, TrackPoint.right_track_pos)
export_splines_to_step(left_spline, right_spline, 'Two Arrows.step')
