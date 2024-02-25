from madcad import vec3, show, Mesh, Web, extrans, mat4, transform, cylinder

# Male Part
cylinder_diameter = 10  # in mm
cylinder_height = 20  # in mm
notch_depth = 1  # in mm
notch_diameter = cylinder_diameter * 1.1  # Slightly larger for the notch

male_part = cylinder(d=cylinder_diameter, h=cylinder_height)
notch = cylinder(d=notch_diameter, h=notch_depth) - cylinder(d=cylinder_diameter, h=notch_depth)
male_part += translate(notch, vec3(0, 0, cylinder_height - notch_depth))

# Female Part
cavity_diameter = cylinder_diameter * 1.02  # Slightly larger for fit
cavity_height = cylinder_height
tab_height = 5  # in mm
tab_thickness = 2  # in mm
tab_protrusion = 1  # in mm

female_part = cylinder(d=cavity_diameter, h=cavity_height, hollow=True, thickness=2)
tab = box(size=(tab_thickness, cavity_diameter, tab_height))
tab_cutout = box(size=(tab_thickness + tab_protrusion*2, cavity_diameter*0.9, tab_height))
tab = tab - translate(tab_cutout, vec3(-tab_protrusion, 0, 0))
tabs = rotate_extrude(tab, angle=360, division=4)  # Create 4 tabs evenly spaced
female_part += translate(tabs, vec3(0, 0, cavity_height - tab_height))