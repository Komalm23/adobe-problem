import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import svgwrite
from svgpathtools import Path, CubicBezier, Line, QuadraticBezier
from shapely.geometry import Point, Polygon, LineString, LinearRing
from scipy import interpolate
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

def polyline_to_bezier(polyline):
    x, y = zip(*polyline)
    tck, u = interpolate.splprep([x, y], k=3, s=0)
    u_new = np.linspace(0, 1, num=len(polyline)*3)
    x_new, y_new = interpolate.splev(u_new, tck)

    bezier_curve = Path()
    for i in range(0, len(x_new) - 3, 3):
        start = complex(x_new[i], y_new[i])
        control1 = complex(x_new[i+1], y_new[i+1])
        control2 = complex(x_new[i+2], y_new[i+2])
        end = complex(x_new[i+3], y_new[i+3])
        bezier_curve.append(CubicBezier(start, control1, control2, end))

    return bezier_curve

def regularize_shape(shape):
    if isinstance(shape, Polygon):
        simplified = shape.simplify(tolerance=0.01)
        exterior = LinearRing(simplified.exterior.coords)
        if is_regular_polygon(exterior):
            return make_regular_polygon(exterior)
        if is_rectangle(simplified):
            return make_rectangle(simplified)
    elif isinstance(shape, LineString):
        if is_straight_line(shape):
            return make_straight_line(shape)
        if is_circular(shape):
            return make_circle_or_ellipse(shape)
    return shape

def is_regular_polygon(ring):
    num_points = len(ring.coords)
    if num_points < 3:
        return False
    centroid = Point(np.mean(ring.xy[0]), np.mean(ring.xy[1]))
    distances = [centroid.distance(Point(ring.xy[0][i], ring.xy[1][i])) for i in range(num_points)]
    if np.std(distances) < 0.01:
        return True
    return False

def make_regular_polygon(ring):
    num_points = len(ring.coords)
    centroid = Point(np.mean(ring.xy[0]), np.mean(ring.xy[1]))
    radius = np.mean([centroid.distance(Point(ring.xy[0][i], ring.xy[1][i])) for i in range(num_points)])
    angle = 360 / num_points
    new_coords = [(centroid.x + radius * np.cos(np.radians(angle * i)),
                   centroid.y + radius * np.sin(np.radians(angle * i))) for i in range(num_points)]
    return Polygon(new_coords)

def is_rectangle(polygon):
    return len(polygon.exterior.coords) == 5 and polygon.is_valid and polygon.is_closed

def make_rectangle(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    return Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])

def is_straight_line(line):
    return line.length > 0 and line.is_simple

def make_straight_line(line):
    coords = list(line.coords)
    return LineString([coords[0], coords[-1]])

def is_circular(line):
    return len(line.coords) > 2 and line.is_simple

def make_circle_or_ellipse(line):
    minx, miny, maxx, maxy = line.bounds
    center = ((minx + maxx) / 2, (miny + maxy) / 2)
    return LineString([center])

def is_reflection_symmetric(shape):
    points = np.array(shape.exterior.coords)
    centroid = np.mean(points, axis=0)
    for angle in range(0, 180, 5):
        axis = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        reflected_points = reflect_points(points, centroid, axis)
        if np.allclose(points, reflected_points, atol=1e-2):
            return True
    return False

def is_rotational_symmetric(shape):
    points = np.array(shape.exterior.coords)
    centroid = np.mean(points, axis=0)
    for n in range(2, 13):
        angle = 360 / n
        rotated_points = rotate_points(points, centroid, angle)
        if np.allclose(points, rotated_points, atol=1e-2):
            return True
    return False

def reflect_points(points, centroid, axis):
    axis = axis / np.linalg.norm(axis)
    points -= centroid
    reflected_points = points - 2 * np.dot(points, axis)[:, np.newaxis] * axis
    return reflected_points + centroid

def rotate_points(points, centroid, angle):
    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])
    points -= centroid
    rotated_points = np.dot(points, rotation_matrix.T)
    return rotated_points + centroid

def complete_curve(polyline):
    points = np.array(polyline)
    gaps = find_gaps(points)
    completed_curve = Path()
    for gap in gaps:
        start, end = gap
        similar_parts = find_similar_parts(points, start, end)
        interpolated_points = interpolate_gap(similar_parts, start, end)
        bezier_segment = polyline_to_bezier(interpolated_points)
        completed_curve.append(bezier_segment)
    return completed_curve

def find_gaps(points):
    gaps = []
    for i in range(len(points) - 1):
        if distance.euclidean(points[i], points[i + 1]) > 5:
            gaps.append((points[i], points[i + 1]))
    return gaps

def find_similar_parts(points, start, end):
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(points)
    start_neighbors = nbrs.kneighbors([start], return_distance=False)
    end_neighbors = nbrs.kneighbors([end], return_distance=False)
    similar_parts = points[np.union1d(start_neighbors, end_neighbors)]
    return similar_parts

def interpolate_gap(similar_parts, start, end):
    similar_start = similar_parts[0]
    similar_end = similar_parts[-1]
    tck, u = interpolate.splprep([similar_start[0], similar_start[1]], k=3, s=0)
    u_new = np.linspace(0, 1, num=10)
    x_new, y_new = interpolate.splev(u_new, tck)
    interpolated_points = list(zip(x_new, y_new))
    return interpolated_points

def rasterize_and_evaluate(svg_path):
    svg = svgwrite.Drawing(svg_path)
    png_data = svg.tostring()
    score = evaluate_rasterization(png_data)
    return score

def evaluate_rasterization(png_data):
    return np.random.random()
