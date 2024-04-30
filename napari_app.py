import napari
from napari.qt import thread_worker
import numpy as np
import dask.array as da
from dask import delayed
import glob
import networkx as nx

from PyQt5.QtCore import Qt

from qtpy.QtWidgets import (
    QWidget, 
    QSizePolicy, 
    QLabel, 
    QGridLayout, 
    QPushButton,
    QProgressBar,
    QSpinBox,
    QCheckBox,
    QComboBox,
)

delayed_load = delayed(np.load)

# N_z, N_y, N_x = np.load('neuron6.npy').shape  # Another way to get the image size?

N_z = 225
N_y = 2400
N_x = 825

print(N_z, N_y, N_x)

img = da.from_delayed(
    delayed_load('neuron6.npy'),
    shape=(N_z, N_y, N_x),
    dtype=float
).rechunk((100, 200, 200))

seg = da.from_delayed(
    delayed_load('pred6.npy') < 8.0,
    shape=(N_z, N_y, N_x),
    dtype=bool
).rechunk((100, 200, 200))

def get_image_chunk(img: da.array, center_loc, chunk_shape) -> da.Array:
    center_loc_array = np.asarray(center_loc).astype(int)
    cz, cy, cx = center_loc_array
    depth, width, length = chunk_shape

    max_z, max_y, max_x = img.shape
    
    start_z = cz - depth // 2
    stop_z = cz + depth // 2
    start_y = cy - width // 2
    stop_y = cy + width // 2
    start_x = cx - length // 2
    stop_x = cx + length // 2

    # Take care of img borders
    start_z = max(start_z, 0)
    start_y = max(start_y, 0)
    start_x = max(start_x, 0)

    stop_z = min(stop_z, max_z)
    stop_y = min(stop_y, max_y)
    stop_x = min(stop_x, max_x)

    img_chunk = img[start_z:stop_z, start_y:stop_y, start_x:stop_x]

    return img_chunk

def get_visible_nodes(img: da.array, center_loc, chunk_shape, path_coordinates) -> da.Array:
    center_loc_array = np.asarray(center_loc).astype(int)
    cz, cy, cx = center_loc_array
    depth, width, length = chunk_shape

    max_z, max_y, max_x = img.shape
    
    start_z = cz - depth // 2
    stop_z = cz + depth // 2
    start_y = cy - width // 2
    stop_y = cy + width // 2
    start_x = cx - length // 2
    stop_x = cx + length // 2

    # Take care of img borders
    start_z = max(start_z, 0)
    start_y = max(start_y, 0)
    start_x = max(start_x, 0)

    stop_z = min(stop_z, max_z)
    stop_y = min(stop_y, max_y)
    stop_x = min(stop_x, max_x)

    path_coordinates_array = np.asarray(path_coordinates).astype(int)
    visible_nodes_filter = (path_coordinates_array[:, 0] >= start_z) & \
        (path_coordinates_array[:, 0] < stop_z) & \
        (path_coordinates_array[:, 1] >= start_y) & \
        (path_coordinates_array[:, 1] < stop_y) & \
        (path_coordinates_array[:, 2] >= start_x) & \
        (path_coordinates_array[:, 2] < stop_x)
    
    visible_nodes = path_coordinates_array[visible_nodes_filter]
    visible_nodes_relative_loc = visible_nodes - center_loc + np.asarray(chunk_shape) // 2

    return visible_nodes_relative_loc


def get_visible_nodes_idx(img: da.array, center_loc, chunk_shape, path_coordinates) -> da.Array:
    center_loc_array = np.asarray(center_loc).astype(int)
    cz, cy, cx = center_loc_array
    depth, width, length = chunk_shape

    max_z, max_y, max_x = img.shape
    
    start_z = cz - depth // 2
    stop_z = cz + depth // 2
    start_y = cy - width // 2
    stop_y = cy + width // 2
    start_x = cx - length // 2
    stop_x = cx + length // 2

    # Take care of img borders
    start_z = max(start_z, 0)
    start_y = max(start_y, 0)
    start_x = max(start_x, 0)

    stop_z = min(stop_z, max_z)
    stop_y = min(stop_y, max_y)
    stop_x = min(stop_x, max_x)

    path_coordinates_array = np.asarray(path_coordinates).astype(int)
    visible_nodes_filter = (path_coordinates_array[:, 0] >= start_z) & \
        (path_coordinates_array[:, 0] < stop_z) & \
        (path_coordinates_array[:, 1] >= start_y) & \
        (path_coordinates_array[:, 1] < stop_y) & \
        (path_coordinates_array[:, 2] >= start_x) & \
        (path_coordinates_array[:, 2] < stop_x)

    return visible_nodes_filter


def get_bbox_location(img: da.array, center_loc, chunk_shape):
    center_loc_array = np.asarray(center_loc).astype(int)
    cz, cy, cx = center_loc_array
    depth, width, length = chunk_shape

    max_z, max_y, max_x = img.shape
    
    start_z = cz - depth // 2
    stop_z = cz + depth // 2
    start_y = cy - width // 2
    stop_y = cy + width // 2
    start_x = cx - length // 2
    stop_x = cx + length // 2

    # Take care of img borders
    start_z = max(start_z, 0)
    start_y = max(start_y, 0)
    start_x = max(start_x, 0)

    stop_z = min(stop_z, max_z)
    stop_y = min(stop_y, max_y)
    stop_x = min(stop_x, max_x)

    return np.array([
        [start_y, start_x],
        [stop_y, stop_x]
    ])

COLORS = [
    'red',
    'green',
    'blue',
    'orange',
    # Add prettier variants
    # ...
]

def get_locs_from_graph(graph, source): 
    locs = []
    for n in nx.dfs_preorder_nodes(graph, source):
        locs.append(graph.nodes[n]["pos"])
    return locs

def load_graph_txt(filename):
     
    G = nx.Graph()
        
    i = 0
    switch = True
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if len(line)==0 and switch:
                switch = False
                continue
            if switch:
                x,y,z = line.split(' ')
                G.add_node(i, pos=(float(x),float(y),float(z)))
                i+=1
            else:
                idx_node1, idx_node2 = line.split(' ')
                G.add_edge(int(idx_node1),int(idx_node2))
    
    return G

class SkeletonWrapper:
    def __init__(self, graph) -> None:
        self._path_coordinates = np.asarray(get_locs_from_graph(graph, source=0))[:, ::-1]
        self.graph = graph
        self.current_idx = 0
        self.edge_color = np.random.choice(COLORS)
        print(f"{self.edge_color=}")
    
    @property
    def path_coordinates(self):
        return self._path_coordinates
    
    @path_coordinates.setter
    def path_coordinates(self, new_coordinates):
        print('Setting - ', new_coordinates)
        self._path_coordinates = new_coordinates
    
    @property
    def path_coordinates_2d(self):
        return self._path_coordinates[:, 1:]
    
    @property
    def num_locs(self):
        return len(self._path_coordinates)
    
    @property
    def shapes_layer(self):
        return self._shapes_layer
    
    @shapes_layer.setter
    def shapes_layer(self, shapes_layer):
        self._shapes_layer = shapes_layer
    
    @property
    def points_layer(self):
        return self._points_layer
    
    @points_layer.setter
    def points_layer(self, points_layer):
        self._points_layer = points_layer
    
    @property
    def minimap_path_layer(self):
        return self._minimap_path_layer
    
    @minimap_path_layer.setter
    def minimap_path_layer(self, minimap_path_layer):
        self._minimap_path_layer = minimap_path_layer

    def _point_move_update_path(self, event):   
        if event.action == "changed":
            self.shapes_layer.data = event.value
    
        elif event.action == "removed":
            self.shapes_layer.data = event.value
    

class NeuronSkeletonWalker(QWidget):
    def __init__(self, img, seg, 
                 skeletons_graph,  # Contains multiple connected components (skeletons)
                 napari_viewer, minimap_viewer) -> None:
        super().__init__()

        self.img = img
        self.seg = seg
        
        self.skeleton_database = {
            idx: SkeletonWrapper(graph) \
                for idx, graph in enumerate(skeletons_graph)
        }

        self.selected_skeleton = self.skeleton_database[0]  # This will get updated quickly anyway

        # Current camera center location
        self.center_loc = self.path_coordinates[self.current_idx]

        self.viewer = napari_viewer
        self.viewer.text_overlay.visible = True

        self.minimap_viewer = minimap_viewer
        self.minimap_viewer.text_overlay.visible = True
        self.minimap_viewer.text_overlay.text = "2D Max Projection in Z"

        # Key bindings
        self.viewer.bind_key('Left', self.move_forward)
        self.viewer.bind_key('Right', self.move_backward)

        ### QT Layout
        grid_layout = QGridLayout()
        grid_layout.setAlignment(Qt.AlignTop)
        self.setLayout(grid_layout)

        # Select a skeleton
        self.cb_skeleton = QComboBox()
        self.cb_skeleton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Add items
        for idx, skeleton in self.skeleton_database.items():
            self.cb_skeleton.addItem(f"{idx}", skeleton)
        
        # Drop down selection changed event
        self.cb_skeleton.currentTextChanged.connect(self._handle_skeleton_changed)

        grid_layout.addWidget(QLabel("Skeleton", self), 0, 0)
        grid_layout.addWidget(self.cb_skeleton, 0, 1)

        # Step forward / backward
        self.forward_btn = QPushButton("Step forward", self)
        self.forward_btn.clicked.connect(self.move_forward)
        grid_layout.addWidget(self.forward_btn, 1, 0)

        self.backward_btn = QPushButton("Step backward", self)
        self.backward_btn.clicked.connect(self.move_backward)
        grid_layout.addWidget(self.backward_btn, 1, 1)

        # Start / Stop button (forward)
        self.play_btn_forward = QPushButton("Move forward", self)
        self.play_btn_forward.clicked.connect(self.toggle_play_forward)
        grid_layout.addWidget(self.play_btn_forward, 2, 0)
        self.running = False

        # Start / Stop button (backward)
        self.play_btn_backward = QPushButton("Move backward", self)
        self.play_btn_backward.clicked.connect(self.toggle_play_backward)
        grid_layout.addWidget(self.play_btn_backward, 2, 1)
        self.running = False

        # Chunk size in X / Y / Z
        grid_layout.addWidget(QLabel("Z"), 3, 0)
        self.z_chunk_spinbox = QSpinBox()
        self.z_chunk_spinbox.setMinimum(1)
        self.z_chunk_spinbox.setMaximum(2000)
        self.z_chunk_spinbox.setValue(20)
        grid_layout.addWidget(self.z_chunk_spinbox, 3, 1)

        grid_layout.addWidget(QLabel("Y"), 4, 0)
        self.y_chunk_spinbox = QSpinBox()
        self.y_chunk_spinbox.setMinimum(1)
        self.y_chunk_spinbox.setMaximum(2000)
        self.y_chunk_spinbox.setValue(100)
        grid_layout.addWidget(self.y_chunk_spinbox, 4, 1)

        grid_layout.addWidget(QLabel("X"), 5, 0)
        self.x_chunk_spinbox = QSpinBox()
        self.x_chunk_spinbox.setMinimum(1)
        self.x_chunk_spinbox.setMaximum(2000)
        self.x_chunk_spinbox.setValue(100)
        grid_layout.addWidget(self.x_chunk_spinbox, 5, 1)

        # Checkbox (view visited / all nodes)
        grid_layout.addWidget(QLabel("View visited nodes", self), 6, 0)
        self.checkbox_view_visited = QCheckBox()
        self.checkbox_view_visited.setEnabled(True)
        self.checkbox_view_visited.setChecked(False)
        self.checkbox_view_visited.stateChanged.connect(self._update_view)
        grid_layout.addWidget(self.checkbox_view_visited, 6, 1)

        # Progress bar
        self.pbar = QProgressBar(self, minimum=0, maximum=1)
        self.pbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(self.pbar, 7, 0, 1, 2)

        # Update the view when the values change in the spinboxes
        self.z_chunk_spinbox.valueChanged.connect(self._update_view)
        self.y_chunk_spinbox.valueChanged.connect(self._update_view)
        self.x_chunk_spinbox.valueChanged.connect(self._update_view)

        # Image layer
        self.image_layer = self.viewer.add_image(
            self.current_image_chunk(),
            multiscale=False,
            contrast_limits = [0, 0.5]
        )

        # Labels layer (hide it by default)
        self.labels_layer = self.viewer.add_labels(
            self.current_labels_chunk(),
            visible=False
        )

        ### SKELETON-WISE LAYERS - colored accordingly! ---------------------
        # # One points layer per skeleton
        for idx, skeleton in self.skeleton_database.items():
            skeleton.points_layer = self.viewer.add_points(
                self.visible_nodes(skeleton),
                face_color=skeleton.edge_color,
                size=1,
                name=f"Nodes {idx}",
            )
            skeleton.points_layer.events.data.connect(skeleton._point_move_update_path)
            skeleton.points_layer.events.data.connect(self._handle_points_data_changed)
        
        # One shapes layer per skeleton
        for idx, skeleton in self.skeleton_database.items():
            skeleton.shapes_layer = self.viewer.add_shapes(
                self.visible_nodes(skeleton),
                shape_type='path',
                edge_color=skeleton.edge_color,
                edge_width=0.2,
                name=f"Skeleton {idx}",
            )

        # One shapes layer per skeleton (minimap)
        for idx, skeleton in self.skeleton_database.items():
            skeleton.minimap_path_layer = self.minimap_viewer.add_shapes(
                skeleton.path_coordinates_2d.copy(),
                shape_type='path',
                edge_color=skeleton.edge_color,
                edge_width=2,
                name=f"Skeleton {idx}",
            )
        ### ------

        # Shapes layer (bounding box) in the minimap viewer
        self.minimap_shapes_layer = self.minimap_viewer.add_shapes(
            self.current_bbox(),
            shape_type='rectangle',
            edge_color='red',
            edge_width=5,
            face_color='transparent',
            name="Current location",
        )
        self.minimap_shapes_layer.mode = 'SELECT'

        # Moving the bounding box updates the 3D view
        self.minimap_shapes_layer.events.set_data.connect(self._minimap_move_bbox)

        self._update_view()

    @property
    def shapes_layer(self):
        return self.selected_skeleton.shapes_layer
    
    @property
    def minimap_path_layer(self):
        return self.selected_skeleton.minimap_path_layer

    @property
    def num_locs(self):
        return self.selected_skeleton.num_locs
    
    @property
    def current_idx(self):
        return self.selected_skeleton.current_idx
    
    @property
    def path_coordinates(self):
        return self.selected_skeleton.path_coordinates
    
    @property
    def path_coordinates_2d(self):
        return self.selected_skeleton.path_coordinates_2d

    def _handle_points_data_changed(self, event):
        global_idx = get_visible_nodes_idx(
            self.img,
            center_loc=self.center_loc,
            chunk_shape=self.chunk_shape,
            path_coordinates=self.path_coordinates
        )
        global_shift = np.where(global_idx)[0][0]

        if event.action == "changed": 
            edited_point_idx = event.data_indices[0]
            new_location = event.value[edited_point_idx] + self.center_loc - np.asarray(self.chunk_shape) // 2

            if np.any(new_location != self.path_coordinates[global_shift + edited_point_idx]):
                print(self.path_coordinates[global_shift + edited_point_idx])
                print(self.selected_skeleton.path_coordinates[global_shift + edited_point_idx])
                self.path_coordinates[global_shift + edited_point_idx] = new_location
                print(self.path_coordinates[global_shift + edited_point_idx])
                print(self.selected_skeleton.path_coordinates[global_shift + edited_point_idx])
                print()

        elif event.action == "removed":
            self.path_coordinates = np.delete(self.path_coordinates, global_shift + event.data_indices, axis=0)

    def _handle_skeleton_changed(self, database_index):
        # print(f"Selected skeleton changed to - {database_index=}")
        self.selected_skeleton = self.skeleton_database[int(database_index)]

        # Update the box center location
        self.center_loc = self.path_coordinates[self.current_idx]

        self._update_view()

    def _minimap_move_bbox(self, e):
        bbox_data = e.source.data[0]
        y0, x0 = bbox_data[0]  # Top left corner
        y1, x1 = bbox_data[2]  # Bottom right corner

        dx = (x1 - x0)
        dy = (y1 - y0)

        center_x = x0 + dx // 2
        center_y = y0 + dy // 2
        center_z = self.path_coordinates[self.current_idx][0]

        self.center_loc = np.array([center_z, center_y, center_x])

        # Disconnect the events to avoid crashing the GUI from recursive calls
        self.y_chunk_spinbox.valueChanged.disconnect(self._update_view)
        self.x_chunk_spinbox.valueChanged.disconnect(self._update_view)
        self.y_chunk_spinbox.valueChanged.connect(self._update_view_without_minimap_bbox)
        self.x_chunk_spinbox.valueChanged.connect(self._update_view_without_minimap_bbox)
        self.y_chunk_spinbox.setValue(int(dy))
        self.x_chunk_spinbox.setValue(int(dx))
        self.y_chunk_spinbox.valueChanged.connect(self._update_view)
        self.x_chunk_spinbox.valueChanged.connect(self._update_view)
        self.y_chunk_spinbox.valueChanged.disconnect(self._update_view_without_minimap_bbox)
        self.x_chunk_spinbox.valueChanged.disconnect(self._update_view_without_minimap_bbox)

        self._update_view_without_minimap_bbox()

    @property
    def chunk_shape(self):
        cz = self.z_chunk_spinbox.value()
        cy = self.y_chunk_spinbox.value()
        cx = self.x_chunk_spinbox.value()
        return (cz, cy, cx)
    
    def _update_view(self):
        self._update_image()
        self._update_labels()
        self._update_shapes()
        self._update_points()        
        self._update_minimap_path()
        self._update_minimap_bbox()
        self._update_overlay()

    def _update_view_without_minimap_bbox(self):
        self._update_image()
        self._update_labels()
        self._update_shapes()
        self._update_points()
        self._update_minimap_path()        
        self._update_overlay()

    def _update_image(self):
        self.image_layer.data = self.current_image_chunk()

    def _update_labels(self):
        self.labels_layer.data = self.current_labels_chunk()

    def _update_shapes(self):
        for skeleton in self.skeleton_database.values():
            
            if self.checkbox_view_visited.isChecked():
                visible_nodes = self.visible_nodes(skeleton)[:(max(skeleton.current_idx, 2))]
            else:
                visible_nodes = self.visible_nodes(skeleton)

            if len(visible_nodes) < 2:  # Quick hack - hide the layer.
                skeleton.shapes_layer.visible = False
                continue
            else:

                skeleton.shapes_layer.visible = True
                skeleton.shapes_layer.data = visible_nodes
                skeleton.shapes_layer.edge_width = 0.2
    
    def _update_points(self):
        for skeleton in self.skeleton_database.values():

            if self.checkbox_view_visited.isChecked():
                visible_nodes = self.visible_nodes(skeleton)[:(max(skeleton.current_idx, 2))]
            else:
                visible_nodes = self.visible_nodes(skeleton)

            if len(visible_nodes) < 2:  # Quick hack - hide the layer.
                skeleton.points_layer.visible = False
                continue
            else:
                skeleton.points_layer.visible = True
                skeleton.points_layer.data = visible_nodes

    def _update_minimap_path(self):
        if self.checkbox_view_visited.isChecked():
            self.minimap_path_layer.data = self.path_coordinates_2d[:(max(self.current_idx, 2))]
        else:
            self.minimap_path_layer.data = self.path_coordinates_2d

    def _update_minimap_bbox(self):
        self.minimap_shapes_layer.data = self.current_bbox()
        self.minimap_shapes_layer.edge_color = self.selected_skeleton.edge_color
    
    def _update_overlay(self):
        self.viewer.text_overlay.text = f"Frame {self.current_idx+1} / {self.num_locs}"

    def current_bbox(self):
        return get_bbox_location(
            self.img,
            center_loc=self.center_loc,
            chunk_shape=self.chunk_shape
        )
    
    # def current_visible_nodes(self):
    def visible_nodes(self, skeleton):
        return get_visible_nodes(
            self.img,
            center_loc=self.center_loc,
            chunk_shape=self.chunk_shape,
            path_coordinates=skeleton.path_coordinates
        )

    def current_image_chunk(self) -> da.array:
        return get_image_chunk(
            self.img,
            center_loc=self.center_loc,
            chunk_shape=self.chunk_shape,
        )
    
    def current_labels_chunk(self) -> da.array:
        return get_image_chunk(
            self.seg,
            center_loc=self.center_loc,
            chunk_shape=self.chunk_shape,
        )
    
    def move_forward(self, *args, **kwargs):
        if self.current_idx + 1 <= self.num_locs-1:
            self.selected_skeleton.current_idx += 1
            self.center_loc = self.path_coordinates[self.current_idx]
        else:
            return

        self._update_view()

    def move_backward(self, *args, **kwargs):
        if self.current_idx - 1 >= 0:
            self.selected_skeleton.current_idx -= 1
            self.center_loc = self.path_coordinates[self.current_idx]
        else:
            return
        
        self._update_view()

    @thread_worker
    def run_animation_foward(self):
        while (self.running is True) & (self.current_idx+1 < self.num_locs):
            self.move_forward()
    
    @thread_worker
    def run_animation_backward(self):
        while (self.running is True) & (self.current_idx > 0):
            self.move_backward()

    def toggle_play_forward(self):
        self.running = not self.running
        if self.running:
            self.play_btn_forward.setText('Stop')
            self.play_btn_backward.setEnabled(False)
            self.pbar.setMaximum(0)  # Start the progress bar
            worker = self.run_animation_foward()
            worker.returned.connect(self.animation_forward_stopped)
            worker.start()
        else:
            print(f"{self.running=}")

    def toggle_play_backward(self):
        self.running = not self.running
        if self.running:
            self.play_btn_backward.setText('Stop')
            self.play_btn_forward.setEnabled(False)
            self.pbar.setMaximum(0)  # Start the progress bar
            worker = self.run_animation_backward()
            worker.returned.connect(self.animation_backward_stopped)
            worker.start()
        else:
            print(f"{self.running=}")

    def animation_forward_stopped(self, return_value=None):
        self.running = False
        self.play_btn_forward.setText('Move forward')
        self.play_btn_backward.setEnabled(True)
        self.pbar.setMaximum(1) # Stop the progress bar
        
    def animation_backward_stopped(self, return_value=None):
        self.running = False
        self.play_btn_backward.setText('Move backward')
        self.play_btn_forward.setEnabled(True)
        self.pbar.setMaximum(1) # Stop the progress bar


minimap_viewer = napari.view_image(da.max(img, axis=0).compute(), contrast_limits=[0, 1], multiscale=False, title='Max Projection')
minimap_viewer.window.qt_viewer.dockLayerControls.setVisible(False)
minimap_viewer.window.qt_viewer.dockLayerList.setVisible(False)

viewer = napari.Viewer(ndisplay=3, title='Image volume subset')
# viewer.window.qt_viewer.dockLayerList.setVisible(False)
# viewer.window.qt_viewer.dockLayerControls.setVisible(False)

skeleton_walker = NeuronSkeletonWalker(
    img, seg, 
    [
        load_graph_txt(file) for file in glob.glob('./DemoData/Hackathon_graphs/*.txt')
    ], 
    viewer, minimap_viewer
)

viewer.window.add_dock_widget(skeleton_walker, name="Neuron walker")

if __name__=='__main__':
    napari.run()