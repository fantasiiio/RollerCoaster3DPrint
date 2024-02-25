import salome
import GEOM
from salome.geom import geomBuilder
import csv

salome.salome_init()
geompy = geomBuilder.New()

class TrackPoint:
    def __init__(self, pos, left):
        self.pos = pos
        self.left = left

    def left_track_pos(self):
        return [self.pos[i] - self.left[i] * 0.5 for i in range(3)]

    def right_track_pos(self):
        return [self.pos[i] + self.left[i] * 0.5 for i in range(3)]

def read_track_data_from_csv(filename):
    track_points = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # Skip header row
        for row in reader:
            pos = (float(row[1]), float(row[2]), float(row[3]))
            left = (float(row[7]), float(row[8]), float(row[9]))
            track_points.append(TrackPoint(pos, left))
    return track_points

def create_spline_from_track_points(track_points, track_selector):
    points = [geompy.MakeVertex(*track_selector(tp)) for tp in track_points]
    spline = geompy.MakeInterpol(points, False)  # False for a non-closed spline
    return spline

def export_splines_to_step(left_spline, right_spline, filename):
    compound = geompy.MakeCompound([left_spline, right_spline])
    geompy.ExportSTEP(compound, filename)

# Example usage
track_points = read_track_data_from_csv('C:\\dev-fg\\RollerCoaster3DPrint\\Two Arrows.csv')
left_spline = create_spline_from_track_points(track_points, TrackPoint.left_track_pos)
right_spline = create_spline_from_track_points(track_points, TrackPoint.right_track_pos)
export_splines_to_step(left_spline, right_spline, 'C:\\dev-fg\\RollerCoaster3DPrint\\Two Arrows.step')

if salome.sg.hasDesktop():
    salome.sg.updateObjBrowser()
