#!/usr/bin/env python
# coding: utf-8

# In[1]:
# Für Hinweise zum Ausführen von Notebooks in Google Colab, siehe:
# https://pytorch.org/tutorials/beginner/colab
# Diese Magics aktivieren sowohl Inline- als auch interaktive Matplotlib-Plots in Notebooks.
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'notebook')


# Main Skript für CGAN 
# =====================
# Dieses Skript dient als Hauptprogramm für das CGAN und nutzt ein GAN-Tutorial von PyTorch als Vorlage.
# Es werden 3D-Meshes als Input verwendet und die Architektur sowie Gewichte können im Laufe der Zeit angepasst werden.

# Einleitung
# ==========
# Hier werden die Anforderungen des CGANs erläutert:
# * CGAN für die Generierung aus spezifischen Formen
# * 3D-Meshes als Input für das Training
# Zudem soll der Code für die Masterarbeit genutzt und weiterentwickelt werden.


# In[3]:
# Import der benötigten Module

import glob                        # Für das Suchen von Dateien anhand von Mustern
import argparse                    # Zur Verarbeitung von Kommandozeilenargumenten (wird hier jedoch nicht weiter genutzt)
import os                          # Für Datei- und Pfadoperationen
import random                      # Für zufallsbasierte Operationen (nicht aktiv verwendet, da numpy genutzt wird)
import sys                         # Für Systemparameter, z. B. für numpy.set_printoptions
import numpy                      # Für numerische Operationen
import traceback                  # Zur Fehlerverfolgung (nicht weiter genutzt)
import logging                    # Für Logging (nicht weiter genutzt)

import torch                      # Hauptpaket von PyTorch für Tensoroperationen und Deep Learning
import torch.nn as nn             # Zum Erstellen von neuronalen Netzwerkschichten
import torch.optim as optim       # Für Optimierungsalgorithmen (z. B. Adam)
from torch import IntTensor       # Wird importiert, aber im Code nicht verwendet
import torch.nn.functional as F   # Für diverse Funktionen wie Aktivierungen und Verlustberechnungen

import torchvision.datasets as dset       # Wird importiert, aber nicht genutzt
import torchvision.transforms as transforms   # Wird importiert, aber nicht genutzt
import torchvision.utils as vutils          # Wird importiert, aber nicht genutzt
from torch.utils.data import Dataset, DataLoader  # Für Datensätze und das Batch-Loading

# Mehrfache Importe von torch (Doppelimporte bleiben erhalten, ohne Änderung)
import torch
from torch_scatter import scatter_mean       # Wird importiert, aber im Code nicht genutzt
from torch_geometric.nn import GCNConv         # Wird importiert, aber nicht verwendet
from torch_geometric.nn import knn_graph       # Wird importiert, aber nicht verwendet

from scipy.spatial import Delaunay            # Für Delaunay-Triangulationen, genutzt in face_generator
from collections import OrderedDict            # Zum Umformen von state_dicts

from pytorch3d.loss import mesh_edge_loss, mesh_normal_consistency  # Beide werden importiert, jedoch wird nur mesh_normal_consistency evtl. genutzt
from pytorch3d.io import load_objs_as_meshes    # Zum Laden von Meshes aus .obj-Dateien (wird im New Data Processing genutzt)
from pytorch3d.ops import sample_points_from_meshes  # Zum Erzeugen von Punktwolken aus Meshes, mehrfach genutzt
from pytorch3d.utils import ico_sphere           # Zum Erzeugen eines geodätischen Netzes (wird in edge_index_generator_sphere definiert, aber nicht aufgerufen)
from pytorch3d.ops import sample_points_from_meshes  # Doppelt importiert, ohne Änderung
from pytorch3d.structures import Meshes          # Zum Erzeugen von Meshes-Objekten
from pytorch3d.io import load_obj, save_obj       # load_obj wird genutzt, save_obj nicht
from pytorch3d.io import IO                       # Wird importiert, aber nicht genutzt
from pytorch3d.datasets import (
    R2N2,
    ShapeNetCore,
    collate_batched_meshes,
    render_cubified_voxels,
)                                               # Mehrere Importe; nur collate_batched_meshes wird in einer Variante evtl. genutzt, andere nicht

import pymeshlab                 # Für Mesh-Vorverarbeitung
from preprocess import find_neighbor  # Zum Finden von Nachbarn in Meshes
from layers import SpatialDescriptor, StructuralDescriptor, MeshConvolution  # Für die Feature-Extraktion in ShapeExtractor
import open3d as o3d             # Für 3D-Punktwolken und Mesh-Rekonstruktion mit Open3D
from open3d.visualization import draw_geometries_with_key_callbacks  # Wird importiert, aber nicht genutzt
import trimesh                   # Zum Laden von OFF-Dateien (in load_off_mesh)
from scipy.spatial import KDTree  # Für schnelle Nachbarschaftssuchen bei Vertex-Dichte-Berechnungen


# In[4]:
# Definition des MeshDataset, eines benutzerdefinierten Datasets zur Vorverarbeitung und zum Laden von Meshes.

class MeshDataset(torch.utils.data.Dataset):
    def __init__(self, root, max_faces, max_vertices=40):
        """
        Dataset-Klasse zum Vorverarbeiten und Laden von Meshes für das Training.
        :param root: Wurzelverzeichnis des Datensatzes (Train-Verzeichnis).
        :param max_faces: Maximale Anzahl von Faces, die pro Mesh verarbeitet werden.
        """
        self.root = root
        self.max_faces = max_faces
        self.max_vertices = max_vertices

        # Liste aller Mesh-Dateien im Trainingsverzeichnis
        self.mesh_files = [
            os.path.join(root, file)
            for file in os.listdir(root)
            if file.endswith('.obj') or file.endswith('.npz') or file.endswith('.off')
        ]

    def __getitem__(self, idx):
        path = self.mesh_files[idx]
        file_name = os.path.basename(path)  # Extrahiere Dateiname, z. B. '17.obj'
        #file_number = int(os.path.splitext(file_name)[0])
        file_number = 0
        file_number = torch.tensor(file_number)
        
        if path.endswith('.stl'):
            vertices, faces = load_stl(path)  # Falls STL-Dateien geladen werden (Funktion load_stl müsste definiert sein)
        elif path.endswith('.npz'):
            # Vorverarbeitete Datei laden
            data = numpy.load(path)
            face = data['faces']
            neighbor_index = data['neighbors']
        else:
            face, neighbor_index, vertices = self.process_mesh(path)
            if face is None:
                # Falls das Mesh zu viele Faces hat, wird das nächste Mesh gewählt.
                return self.__getitem__((idx + 1) % len(self.mesh_files))          
        num_point = len(face)
        num_vertices = vertices.shape[0]
       
        # Falls die Anzahl der Faces oder Vertices kleiner als das Maximum ist, wird Padding durchgeführt.
        if num_point < self.max_faces or num_vertices < self.max_vertices:
            random_indices_face = numpy.random.randint(0, num_point, self.max_faces - num_point) if num_point < self.max_faces else []
            random_indices_vertices = torch.randint(0, num_vertices, (self.max_vertices - num_vertices,), device=vertices.device) if num_vertices < self.max_vertices else []
            
            # Face-Padding
            if num_point < self.max_faces:
                fill_face = face[random_indices_face]
                fill_neighbor_index = neighbor_index[random_indices_face]
                face = numpy.concatenate((face, fill_face))
                neighbor_index = numpy.concatenate((neighbor_index, fill_neighbor_index))

            # Vertex-Padding
            if num_vertices < self.max_vertices:
               random_vertices = vertices[random_indices_vertices]
               vertices = torch.tensor(vertices, dtype=torch.float32)
               random_vertices = torch.tensor(random_vertices, dtype=torch.float32)
               vertices_t = torch.cat([vertices, random_vertices], dim=0)
            else:
                vertices_t = torch.tensor(vertices, dtype=torch.float32)
        else:
             vertices_t = torch.tensor(vertices, dtype=torch.float32)

        # Konvertiere Face- und Neighbor-Daten in PyTorch-Tensoren
        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()

        # Die Face-Daten werden so permutiert, dass die ersten 3 Zeilen die Face-Zentren, die nächsten 9 die Ecken und danach die Normalen enthalten.
        face = face.permute(1, 0).contiguous()  # (features, num_faces)
        centers, corners, normals = face[:3], face[3:12], face[12:]
        # Zentriere die Ecken relativ zu den Face-Zentren
        corners = corners - torch.cat([centers, centers, centers], 0)

        # Berechne den Krümmungswert (curve_score) für die Faces
        curve_score = torch.tensor(comp_curve_score(normals, neighbor_index)).to(device)
        centers = centers.to(device)
        corners = corners.to(device)
        normals = normals.to(device)
        neighbor_index = neighbor_index.to(device)
        
        return centers, corners, normals, neighbor_index, vertices_t, file_number, curve_score

    def __len__(self):
        return len(self.mesh_files)

    def process_mesh(self, path):
        """
        Vorverarbeitet eine einzelne Mesh-Datei.
        :param path: Pfad zur Mesh-Datei.
        :return: Vorverarbeitete Face- und Neighbor_Index-Arrays sowie die ursprünglichen Vertices.
        """
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(path)
        mesh = ms.current_mesh()       
        vertices = mesh.vertex_matrix()
        faces = mesh.face_matrix()
        obj_verts = vertices

        if faces.shape[0] > self.max_faces:
            # Falls das Mesh zu viele Faces hat, wird None zurückgegeben.
            return None, None, None

        # Normalisiere das Mesh, indem es zentriert und skaliert wird.
        center = (numpy.max(vertices, 0) + numpy.min(vertices, 0)) / 2
        vertices -= center
        max_len = numpy.max(vertices[:, 0]**2 + vertices[:, 1]**2 + vertices[:, 2]**2)
        vertices /= numpy.sqrt(max_len)

        # Berechne die Face-Normalen
        ms.clear()
        ms.add_mesh(pymeshlab.Mesh(vertices, faces))
        face_normals = ms.current_mesh().face_normal_matrix()

        # Berechne Face-Zentren und Ecken
        faces_contain_this_vertex = [set() for _ in range(len(vertices))]
        centers = []
        corners = []
        for i, face in enumerate(faces):
            v1, v2, v3 = face
            x1, y1, z1 = vertices[v1]
            x2, y2, z2 = vertices[v2]
            x3, y3, z3 = vertices[v3]
            centers.append([(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3])
            corners.append([x1, y1, z1, x2, y2, z2, x3, y3, z3])
            faces_contain_this_vertex[v1].add(i)
            faces_contain_this_vertex[v2].add(i)
            faces_contain_this_vertex[v3].add(i)

        # Bestimme Nachbarn für jedes Face
        neighbors = []
        for i, face in enumerate(faces):
            v1, v2, v3 = face
            n1 = find_neighbor(faces, faces_contain_this_vertex, v1, v2, i)
            n2 = find_neighbor(faces, faces_contain_this_vertex, v2, v3, i)
            n3 = find_neighbor(faces, faces_contain_this_vertex, v3, v1, i)
            neighbors.append([n1, n2, n3])

        # Konvertiere die gesammelten Daten in NumPy-Arrays
        centers = numpy.array(centers)
        corners = numpy.array(corners)
        faces = numpy.concatenate([centers, corners, face_normals], axis=1)
        neighbors = numpy.array(neighbors)
        
        return faces, neighbors, obj_verts

# Collate-Funktion zur Batch-Zusammenstellung
def collate_fn1(batch, max_faces=1700):
    centers, corners, normals, neighbor_indices, vertices, file_number, curve_score = zip(*batch)

    # Staple alle Elemente zu Tensoren
    centers = torch.stack(centers)
    corners = torch.stack(corners)
    normals = torch.stack(normals)
    neighbor_indices = torch.stack(neighbor_indices)
    vertices = torch.stack(vertices)
    file_numbers = torch.stack(file_number)
    curve_score = torch.stack(curve_score)
    
    return centers, corners, normals, neighbor_indices, vertices, file_numbers, curve_score

# Funktion zum Laden eines Meshes aus einer OFF-Datei mithilfe von Trimesh
def load_off_mesh(path):
    """
    Lädt ein Mesh aus einer OFF-Datei.
    """
    mesh = trimesh.load_mesh(path, file_type='off')
    vertices = numpy.array(mesh.vertices)
    faces = numpy.array(mesh.faces)
    return vertices, faces

# Funktion zum Auffüllen (Padding) von Vertices, falls zu wenige vorhanden sind
def pad_verts(vertices, target_size):
    """
    Füllt die Vertices mit zufällig existierenden Werten auf.
    :param vertices: Tensor der Form (batch_size, num_vertices, 3)
    :param target_size: Zielanzahl der Vertices
    :return: Gepaddeter Tensor der Form (batch_size, target_size, 3)
    """
    batch_size, current_size, dim = vertices.size()
    if current_size >= target_size:
        return vertices

    # Anzahl der zusätzlich benötigten Vertices
    pad_size = target_size - current_size

    # Zufälliges Sampling existierender Vertices
    random_indices = torch.randint(0, current_size, (batch_size, pad_size), device=vertices.device)
    random_vertices = torch.stack([vertices[b, random_indices[b]] for b in range(batch_size)], dim=0)

    # Bestehende Vertices mit den gepaddeten kombinieren
    padded_vertices = torch.cat([vertices, random_vertices], dim=1)
    return padded_vertices

# Funktion zum Parsen einer ShapesList.txt, um eine Zuordnung von Nummern zu Shape-Namen zu erstellen
def parse_shapes_list(file_path):
    """
    Parst ShapesList.txt und erstellt ein Dictionary, das Nummern zu Shape-Namen zuordnet.
    :param file_path: Pfad zu ShapesList.txt
    :return: Dictionary {int: str}
    """
    shape_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Zeilen im Format "1 = ShapeName" parsen
            if '=' in line:
                number, shape = line.split('=')
                shape_dict[int(number.strip())] = shape.strip()
    return shape_dict

# Funktion zur Berechnung des Krümmungswerts (curve_score) basierend auf den Normalen und Nachbarn
def comp_curve_score(normals, neighbor_index):
    """
    Berechnet Krümmungswerte für Faces anhand des Winkels zwischen benachbarten Face-Normalen.
    :param normals: Tensor der Form (num_faces, 3) - Normalen für jedes Face
    :param neighbor_index: Tensor der Form (num_faces, k_neighbors) - Nachbar-Indizes für jedes Face
    :return: Tensor der Form (num_faces,) - Krümmungswerte für jedes Face
    """
    # Nehme die Normalen und permutiere, um mit den Nachbarn zu arbeiten
    normals = normals.permute(1,0)
    neighbor_normals = normals[neighbor_index]  # Form: (num_faces, k_neighbors, 3)

    # Berechne das Skalarprodukt zwischen den Face-Normalen und ihren Nachbarn
    dot_products = torch.sum(normals.unsqueeze(1) * neighbor_normals, dim=-1)  # (num_faces, k_neighbors)

    # Clamp die Skalarprodukte auf [-1, 1], um numerische Probleme bei acos zu vermeiden
    dot_products = torch.clamp(dot_products, -1.0, 1.0)

    # Berechne die Winkel (in Radiant) zwischen den Normalen
    angles = torch.acos(dot_products)  # (num_faces, k_neighbors)

    # Durchschnittswinkel als Krümmungswert für jedes Face
    scores = torch.mean(angles, dim=-1)  # (num_faces,)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores



# New Data Processing
# ===================
# Hier werden Open3D-Operationen durchgeführt, um aus einem geladenen Mesh eine Punktwolke zu erstellen
# und mittels Ball Pivoting Algorithmus (BPA) ein trianguliertes Mesh zu rekonstruieren.

# In[ ]:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mesh = load_objs_as_meshes(["7.obj"], device=device)
point_cloud = sample_points_from_meshes(mesh, 500).squeeze(0).cpu().numpy()

# Konvertiere in ein Open3D-Punktwolken-Objekt
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

pcd.estimate_normals()

# Ball Pivoting Algorithmus anwenden
radii = [0.005, 0.01, 0.02]  # Radii anpassen für eine bessere Rekonstruktion
mesh_bpa = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii)
)

# Speichere und visualisiere das rekonstruierte Mesh
o3d.io.write_triangle_mesh("bpa_triangulated_mesh.obj", mesh_bpa)
o3d.visualization.draw_geometries([mesh_bpa])


# Face-Generator
# ==============
# Konvertiert eine Batch von Punktwolken in Meshes mittels Delaunay-Triangulation.

# In[5]:
def face_generator(point_clouds):
    """
    Konvertiere eine Batch von Punktwolken in eine Batch von PyTorch3D-Meshes.
    
    Args:
        point_clouds: Tensor der Form (batch_size, num_points, 3), die Punktwolken.
    
    Returns:
        meshes: PyTorch3D Meshes-Objekt mit dem Batch an Meshes.
    """
    # print(point_clouds.size())
    batch_size = point_clouds.size(0)
    verts_list = []  # Liste für Vertices (pro Mesh)
    faces_list = []  # Liste für Faces (pro Mesh)
    
    for i in range(batch_size):
        # Konvertiere die Punktwolke zu NumPy für die Delaunay-Triangulation
        points = point_clouds[i].cpu().detach().numpy()
        
        # Führe Delaunay-Triangulation durch
        tri = Delaunay(points)

        # Extrahiere Vertices und Faces aus der Triangulation
        verts = torch.tensor(tri.points, dtype=torch.float32).to(point_clouds.device)  # Form: (num_vertices, 3)
        faces = torch.tensor(tri.convex_hull, dtype=torch.int64).to(point_clouds.device)  # Form: (num_faces, 3)

        # Filtere ungültige Faces (mit -1) heraus
        valid_faces = faces[(faces >= 0).all(dim=1)]  # Entferne Faces mit -1

        # Speichere Vertices und gültige Faces in den Listen
        verts_list.append(verts)
        faces_list.append(valid_faces)
        
    # Erstelle ein Meshes-Objekt aus den gesammelten Vertices und Faces
    meshes = Meshes(verts=verts_list, faces=faces_list).to(point_clouds.device)
    return {"mesh": meshes, "verts": verts_list, "faces": faces_list}


# In[ ]:
# (Leere Zelle)


# Edge Index Generator (Sphere)
# =============================
# Diese Funktion erzeugt einen Edge-Index für eine kugelförmige Oberfläche, wird hier jedoch nicht im Trainingsloop aufgerufen.

# In[6]:
def edge_index_generator_sphere():
    # Erstelle eine kugelförmige Oberfläche mit 100 Vertices
    mesh = ico_sphere(level=2)  # Erzeugt ein geodätisches Netz mit ca. 100 Vertices

    # Extrahiere Vertices und Faces
    verts = mesh.verts_list()[0]  # (N, 3) - Positionen der Vertices
    faces = mesh.faces_list()[0]  # (F, 3) - Indizes der Dreiecke

    # Kanten aus den Faces extrahieren
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)

    # Kanten sortieren und Duplikate entfernen
    edges = torch.sort(edges, dim=1)[0]
    edges = torch.unique(edges, dim=0)

    # Erstelle den edge_index
    edge_index_g = edges.T  # Form: (2, num_edges)
    edge_index_g = edge_index_g.to(device)
   
    # print(edge_index_g)
    return edge_index_g


# Edge Index Generator
# ====================
# Diese Funktion erzeugt einen Edge-Index aus den übergebenen Face-Daten.

# In[7]:
def edge_index_generator(faces):
   edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
   edges = torch.sort(edges, dim=1)[0]  # Sicherstellen einer konsistenten Kantenreihenfolge
   edge_index = torch.unique(edges, dim=0).T  # Form: (2, num_edges)
   edge_index = edge_index.to(torch.int64)
   return edge_index


# Shape Extractor
# ===============
# Extrahiert Form-Features aus den Eingabedaten mittels spezialisierter Layer.
# In[8]:
class ShapeExtractor(nn.Module):
    def __init__(self, cfg):
        super(ShapeExtractor, self).__init__()
        self.spatial_descriptor = SpatialDescriptor()
        self.structural_descriptor = StructuralDescriptor(cfg['structural_descriptor'])
        self.mesh_conv1 = MeshConvolution(cfg['mesh_convolution'], 64, 131, 96, 96)
        self.mesh_conv2 = MeshConvolution(cfg['mesh_convolution'], 96, 96, 96, 64)
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(160, 96, 1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
        )
        self.concat_mlp = nn.Sequential(
            nn.Conv1d(96 + 96 + 96, 128, 1), 
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
    
    def forward(self, centers, corners, normals, neighbor_index, curvature_scores):
        spatial_fea0 = self.spatial_descriptor(centers)
        structural_fea0 = self.structural_descriptor(corners, normals, neighbor_index)

        spatial_fea1, structural_fea1 = self.mesh_conv1(spatial_fea0, structural_fea0, neighbor_index)
        spatial_fea2, structural_fea2 = self.mesh_conv2(spatial_fea1, structural_fea1, neighbor_index)
        spatial_fea3 = self.fusion_mlp(torch.cat([spatial_fea2, structural_fea2], 1))

        combined_fea = torch.cat([spatial_fea1, spatial_fea2, spatial_fea3], 1)  # b, c, n

        curvature_weight = curvature_scores.unsqueeze(1).expand_as(combined_fea)    
        combined_fea_with_scores = combined_fea 
        
        # Aggregiere globale Features aus den kombinierten lokalen Features
        aggregated_fea = self.concat_mlp(combined_fea_with_scores)  # (batch_size, 512, num_faces)
        global_fea = torch.max(aggregated_fea, dim=2)[0]  # Globales Feature (batch_size, 512)
        
        return global_fea


# Generator
# =========
# Der Generator erzeugt Mesh-Vertices basierend auf extrahierten globalen Features.
# In[9]:
class MeshGenerator(nn.Module):
    def __init__(self, cfg):
        super(MeshGenerator, self).__init__()
        self.shape_extractor = ShapeExtractor(cfg)
        self.vertex_decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, cfg['num_vertices'] * 3)  # Ausgabeform: (batch_size, num_vertices * 3)
        )
    
    def forward(self, centers, corners, normals, neighbor_index, curvature_scores):
        # Extrahiere globale Features mit dem ShapeExtractor
        global_fea = self.shape_extractor(centers, corners, normals, neighbor_index, curvature_scores)
        # Decodiere die globalen Features in Vertex-Positionen
        generated_vertices = self.vertex_decoder(global_fea)
        generated_vertices = generated_vertices.view(generated_vertices.size(0), -1, 3)  # Reshape zu (batch_size, num_vertices, 3)
        generated_vertices = scale_to_unit(generated_vertices)  # Skaliere die Vertices in den Bereich [-1, 1]
        
        return generated_vertices

# Hilfsfunktion zum Skalieren der Vertices in den Bereich [-1, 1]
def scale_to_unit(vertices):
    """
    Skaliert Vertices so, dass sie den Bereich [-1, 1] abdecken, wobei die Form erhalten bleibt.
    :param vertices: Tensor der Form (batch_size, num_vertices, 3).
    :return: Skalierte Vertices.
    """
    # Schritt 1: Zentriere die Vertices um den Ursprung
    centroid = torch.mean(vertices, dim=1, keepdim=True)  # Berechne den Schwerpunkt (batch_size, 1, 3)
    centered_vertices = vertices - centroid               # Zentriere die Vertices (batch_size, num_vertices, 3)

    # Schritt 2: Finde den maximalen absoluten Wert entlang einer Achse
    max_abs = torch.amax(torch.abs(centered_vertices), dim=(1, 2), keepdim=True)[0]  # (batch_size, 1, 1)

    # Schritt 3: Skaliere die Vertices in den Bereich [-1, 1]
    scaled_vertices = centered_vertices / max_abs

    return scaled_vertices


# Diskriminator
# ============
# Das Diskriminator-Netzwerk klassifiziert, ob ein Mesh echt oder generiert ist.
# In[10]:
class MeshDiscriminator(nn.Module):
    def __init__(self, cfg):
        super(MeshDiscriminator, self).__init__()
        
        # Convolutional Layers zur Extraktion lokaler Features aus den Vertex-Positionen
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)

        # Batch-Normalisierung für Stabilität
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)

        # Fully Connected Layers für die Klassifikation
        self.fc1 = nn.Linear(128 * 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        # Erwarteter Input: (batch_size, num_vertices, 3)
        
        x = x.permute(0, 2, 1)  # Ändere die Dimensionen zu (batch_size, 3, num_vertices) für Conv1d
    
        # Wende die Convolutional Layers mit ReLU-Aktivierung und Batch-Normalisierung an
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Verwende Adaptive Average Pooling, um eine feste Ausgabelänge (64) zu erhalten
        x = F.adaptive_avg_pool1d(x, 64)
        
        # Flatten für die Fully Connected Layers
        x = x.view(x.shape[0], -1)
        
        # Fully Connected Layers mit ReLU-Aktivierung, abschließend Sigmoid für die Klassifikation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid für binäre Klassifikation

        return x.squeeze()


# Loss-Functions
# ==============
# Hier werden spezielle Verlustfunktionen definiert.

# In[11]:
def chamfer_distance1(x, y):
    # Berechnet den Chamfer-Loss zwischen zwei Punktwolken
    x_exp = x.unsqueeze(2)  # (batch_size, num_points, 1, 3)
    y_exp = y.unsqueeze(1)  # (batch_size, 1, num_points, 3)
    dist = torch.norm(x_exp - y_exp, dim=-1)  # Paarweise Distanzen
    min_dist_x, _ = torch.min(dist, dim=2)
    min_dist_y, _ = torch.min(dist, dim=1)
    return torch.mean(min_dist_x) + torch.mean(min_dist_y)

def compute_vertex_density(vertices, k=10):
    # Berechnet die Dichte der Vertices mithilfe eines KDTree
    tree = KDTree(vertices)
    density_scores = numpy.zeros(len(vertices))

    for i, vertex in enumerate(vertices):
        dists, idx = tree.query(vertex, k=k+1)  # +1, da der Punkt selbst enthalten ist
        density_scores[i] = k / (numpy.mean(dists[1:]) + 1e-8)  # Durchschnittliche Entfernung ohne den Punkt selbst

    return density_scores

def curve_loss(generated_vertices, corners, curvature_scores):
    """
    Bestraft den Generator, wenn Vertices nicht in Regionen mit hoher Krümmung platziert werden.
    :param generated_vertices: Tensor der Form (batch_size, num_generated_vertices, 3)
    :param corners: Tensor der Form (batch_size, num_faces, 9) - echte Ecken
    :param curvature_scores: Tensor der Form (batch_size, num_faces) - echte Krümmungswerte
    :return: Loss-Wert
    """
    corners = corners.permute(0,2,1)
    # Berechne die Distanzen zwischen generierten Vertices und Ecken
    generated_vertices = generated_vertices.unsqueeze(2)  # (batch_size, num_generated_vertices, 1, 3)
    corners = corners.view(corners.size(0), corners.size(1), 3, 3)  # (batch_size, num_faces, 3, 3)
    distances = torch.norm(generated_vertices - corners, dim=-1)  # (batch_size, num_generated_vertices, num_faces, 3)

    # Gewichte die Distanzen mit den Krümmungswerten
    weighted_distances = distances * curvature_scores.unsqueeze(1).unsqueeze(-1)  # (batch_size, num_generated_vertices, num_faces, 3)

    # Minimiere die gewichteten Distanzen
    return weighted_distances.mean()
    
def map_vertices_to_faces(generated_vertices, face_centers):
    """
    Findet für jedes generierte Vertex das nächstgelegene Face anhand der Mittelpunkte.
    
    :param generated_vertices: (N, 3) numpy array der generierten Vertex-Positionen
    :param face_centers: (M, 3) numpy array mit den Mittelpunkten der echten Faces
    :return: (N,) numpy array, das für jedes generierte Vertex den Index des nächstgelegenen Faces enthält
    """
    face_centers = face_centers.T
    tree = KDTree(face_centers)  # KDTree für schnellen Nachbarschaftsvergleich
    _, closest_face_indices = tree.query(generated_vertices)  # Finde das nächstgelegene Face für jedes Vertex

    return closest_face_indices
    
def compute_face_vertex_density(generated_vertices, face_centers, num_faces):
    """
    Berechnet die Vertex-Dichte für jedes echte Face basierend auf den generierten Vertices.
    
    :param generated_vertices: (N, 3) numpy array mit generierten Vertex-Positionen
    :param face_centers: (M, 3) numpy array mit den Mittelpunkten der echten Faces
    :param num_faces: Anzahl der echten Faces
    :return: (M,) numpy array mit der Anzahl der generierten Vertices pro Face
    """
    face_density = numpy.zeros(num_faces)
    # Bestimme, welches Face zu jedem generierten Vertex gehört
    closest_faces = map_vertices_to_faces(generated_vertices, face_centers)
    # Zähle, wie viele Vertices pro Face liegen
    for face_idx in closest_faces:
        face_density[face_idx] += 1
    # Normalisiere die Dichte auf einen Bereich von 0 bis 1
    face_density = (face_density - numpy.min(face_density)) / (numpy.max(face_density) - numpy.min(face_density) + 1e-8)
    
    return face_density

def curvature_density_loss(generated_vertices, face_centers, real_curvature_scores):
    """
    Bestraft falsche Vertex-Verteilungen basierend auf dem echten Krümmungs-Score der Faces.
    
    :param generated_vertices: (B, N, 3) Tensor mit den generierten Vertex-Positionen
    :param face_centers: (B, M, 3) Tensor mit den Mittelpunkten der echten Faces
    :param real_curvature_scores: (B, M) Tensor mit den echten Krümmungswerten der Faces
    """
    batch_size = generated_vertices.shape[0]
    loss = 0.0

    for i in range(batch_size):
        # Konvertiere in NumPy für Berechnungen mit KDTree
        verts = generated_vertices[i].detach().cpu().numpy()
        centers = face_centers[i].detach().cpu().numpy()
        curv_scores = real_curvature_scores[i]
        # Berechne die Face-Dichte basierend auf den generierten Vertices
        face_vertex_density = compute_face_vertex_density(verts, centers, centers.shape[1])
        # Konvertiere in einen PyTorch Tensor
        face_vertex_density = torch.tensor(face_vertex_density, dtype=torch.float32, device=generated_vertices.device)
        # Berechne den Fehler als Differenz zwischen generierter Vertex-Dichte und echtem Krümmungswert
        density_diff = torch.abs(face_vertex_density - curv_scores)
        # Mean Squared Error als Bestrafung
        loss += torch.mean(density_diff ** 2)
    
    return loss / batch_size  # Durchschnitt über den Batch


# Training
# ==========
# Festlegung der Trainingsparameter, Initialisierung von Generator und Diskriminator und Ausführung des Trainingsloops.

# In[21]:
latent_dim = 128   # Dimension des latenten Vektors
num_epochs = 1000  # Anzahl der Trainingsepochen
batch_size = 23    # Batch-Größe
N = 1024           # (Definiert, aber nicht verwendet)
lr_g = 1e-4        # Lernrate für den Generator
lr_d = 1e-5        # Lernrate für den Diskriminator
regulator = 0.2    # (Definiert, aber nicht genutzt)
feature_dim = 512  # Dimension der extrahierten Features (z. B. von DGCNN)
num_verts = 850    # Anzahl der Vertices im generierten Mesh
num_faces = 1700   # Maximale Anzahl an Faces
num_workers = 4    # (Definiert, aber nicht verwendet)

# Hinweis: num_vertices sollte möglichst nah an realen Objekten liegen.
# (Zum Testen: num_vertices = Vertices in Sphere, für Testing)
# Initialisierung von Generator und Diskriminator
cfg = {
    'structural_descriptor': {
        'num_kernel': 64,
        'sigma': 0.2
    },
    'mesh_convolution': {
        'aggregation_method': 'Max'
    },
    'num_vertices': num_verts  # Anzahl der Vertices im Input-Mesh
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = MeshGenerator(cfg)
discriminator = MeshDiscriminator(cfg)

generator = generator.to(device)
discriminator = discriminator.to(device)

# Optimizer für Generator und Diskriminator mit Adam
optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

# Funktion zum Initialisieren der Gewichte (Xavier-Initialisierung)
def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)



# Laden eines state_dict und Entfernen des "module."-Präfixes
state_dict = torch.load("MeshNet_best_9192.pkl")
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("module.", "")  # Entferne "module."-Präfix
    new_state_dict[new_key] = v

# batch_indices wird definiert, aber nicht weiter genutzt
batch_indices = torch.arange(batch_size).repeat_interleave(num_verts).to(device)

# Initialisiere das MeshDataset und den DataLoader
meshDataset = MeshDataset(root ="C:/Users/Ben/Desktop/MA_PyTorch/newModelData/goodTopo", max_faces=num_faces, max_vertices=num_verts)
meshDataloader = DataLoader(dataset=meshDataset, batch_size=batch_size, shuffle="true", collate_fn=collate_fn1)

# shapes_dict wird aus einer ShapesList.txt geparst, aber danach nicht verwendet
shapes_dict = parse_shapes_list("ShapesList.txt")

print("Starting training...")

# Trainingsloop
for epoch in range(num_epochs):
    
    # Hole einen Batch aus dem MeshDataloader
    centers, corners, normals, neighbor_index, vertices, file_numbers, curve_score = next(iter(meshDataloader))    

    centers = centers.to(dtype=torch.float32)
    corners = corners.to(dtype=torch.float32)
    normals = normals.to(dtype=torch.float32)

    # -----------------
    # Training des Diskriminators
    # -----------------
    optimizer_D.zero_grad()
   
    vertices = vertices.to(device) 
    real_pred = discriminator(vertices)  # Diskriminator gibt eine Bewertung für reale Meshes
    real_labels = torch.ones_like((real_pred)).to(device)  # Label 1 für reale Daten
    real_loss = F.binary_cross_entropy(real_pred, real_labels).to(device)
    
    # Schritt 2: Generiere Daten und verarbeite sie durch den Diskriminator
    generated_verts = generator(centers, corners, normals, neighbor_index, curve_score)  # Generiere Mesh-Vertices
    generated_verts = generated_verts.to(device) 
    fake_pred = discriminator(generated_verts)  # Bewertung der generierten Daten durch den Diskriminator
    fake_labels = torch.zeros_like(fake_pred).to(device)  # Label 0 für generierte Daten
    fake_loss = F.binary_cross_entropy(fake_pred, fake_labels).to(device)
    chamfer_loss = chamfer_distance1(generated_verts, vertices)  # Berechne Chamfer-Loss zwischen generierten und realen Vertices
    chamfer_loss_d = fake_loss + chamfer_loss 

    # Diskriminator-Loss als Summe von real_loss, fake_loss und Chamfer-Loss
    adv_d = real_loss + fake_loss
    d_loss = real_loss + fake_loss + chamfer_loss_d 
    d_loss.backward()
    optimizer_D.step()
    
    # -----------------
    # Training des Generators
    # -----------------
    optimizer_G.zero_grad()

    generated_verts = generator(centers, corners, normals, neighbor_index, curve_score).to(device)   
    generated_meshes = face_generator(generated_verts)  # Erzeuge Meshes aus den generierten Vertices

    real_meshes_cd = face_generator(vertices)  # Erzeuge Meshes aus den realen Vertices
     
    # Diskriminator-Output für die generierten Vertices
    generated_verts = generated_verts.to(device) 
    fake_pred = discriminator(generated_verts)
    
    adv_loss = F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred)).to(device)
    
    fake_point_clouds = sample_points_from_meshes(generated_meshes["mesh"], num_samples=1024).to(device)
    real_point_clouds = sample_points_from_meshes(real_meshes_cd["mesh"],num_samples=1024).to(device)

    curve_loss = curvature_density_loss(generated_verts, centers, curve_score)  # Berechne den Verlust basierend auf der Krümmungsdichte
    chamfer_loss = chamfer_distance1(generated_verts,vertices)  # Berechne Chamfer-Loss zwischen generierten und realen Vertices

    # Gesamter Generator-Loss als Summe aus adversarial loss, Chamfer-Loss und Kurven-Loss
    g_loss = adv_loss + chamfer_loss + curve_loss
    g_loss.backward()
    optimizer_G.step()

    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss D: {d_loss.item():.4f}, Loss Adv_d: {adv_d.item():.4f} ------ Loss G: {g_loss.item():.4f}, Curve Loss: {curve_loss.item():.4f}, Chamfer Loss: {chamfer_loss.item():.4f}")



