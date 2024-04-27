import napari
from napari.qt import thread_worker
import numpy as np
import dask.array as da
from dask import delayed

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
)

delayed_load = delayed(np.load)


# N_z, N_y, N_x = np.load('neuron6.npy').shape  # Another way to get the image size?
N_z = 225
N_y = 2400
N_x = 825

path_coordinates = np.load('DemoData/demo_locations2.npy')[:, ::-1]


# load data
img = da.from_delayed(
    delayed_load('mallory/neuron6.npy'),
    shape=(N_z, N_y, N_x),
    dtype=float
).rechunk((100, 200, 200))

seg = da.from_delayed(
    delayed_load('mallory/pred6.npy') < 8.0,
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


class NeuronSkeletonWalker(QWidget):
    def __init__(self, img, seg, path_coordinates, napari_viewer, minimap_viewer) -> None:
        super().__init__()

        self.path_coordinates = path_coordinates
        self.num_locs = len(path_coordinates)
        self.current_idx = 0

        self.center_loc = self.path_coordinates[self.current_idx]

        self.img = img
        self.seg = seg

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

        # Step forward / backward
        self.forward_btn = QPushButton("Step forward", self)
        self.forward_btn.clicked.connect(self.move_forward)
        grid_layout.addWidget(self.forward_btn, 0, 0)

        self.backward_btn = QPushButton("Step backward", self)
        self.backward_btn.clicked.connect(self.move_backward)
        grid_layout.addWidget(self.backward_btn, 0, 1)

        # Start / Stop button (forward)
        self.play_btn_forward = QPushButton("Move forward", self)
        self.play_btn_forward.clicked.connect(self.toggle_play_forward)
        grid_layout.addWidget(self.play_btn_forward, 1, 0)#, 1, 2)
        self.running = False

        # Start / Stop button (backward)
        self.play_btn_backward = QPushButton("Move backward", self)
        self.play_btn_backward.clicked.connect(self.toggle_play_backward)
        grid_layout.addWidget(self.play_btn_backward, 1, 1)#, 1, 2)
        self.running = False

        # Chunk size in X / Y / Z
        grid_layout.addWidget(QLabel("Z"), 2, 0)
        self.z_chunk_spinbox = QSpinBox()
        self.z_chunk_spinbox.setMinimum(1)
        self.z_chunk_spinbox.setMaximum(2000)
        self.z_chunk_spinbox.setValue(20)
        grid_layout.addWidget(self.z_chunk_spinbox, 2, 1)

        grid_layout.addWidget(QLabel("Y"), 3, 0)
        self.y_chunk_spinbox = QSpinBox()
        self.y_chunk_spinbox.setMinimum(1)
        self.y_chunk_spinbox.setMaximum(2000)
        self.y_chunk_spinbox.setValue(100)
        grid_layout.addWidget(self.y_chunk_spinbox, 3, 1)

        grid_layout.addWidget(QLabel("X"), 4, 0)
        self.x_chunk_spinbox = QSpinBox()
        self.x_chunk_spinbox.setMinimum(1)
        self.x_chunk_spinbox.setMaximum(2000)
        self.x_chunk_spinbox.setValue(100)
        grid_layout.addWidget(self.x_chunk_spinbox, 4, 1)

        # Checkbox (view visited / all nodes)
        grid_layout.addWidget(QLabel("View visited nodes", self), 5, 0)
        self.checkbox_view_visited = QCheckBox()
        self.checkbox_view_visited.setEnabled(True)
        self.checkbox_view_visited.setChecked(True)
        self.checkbox_view_visited.stateChanged.connect(self._update_view)
        grid_layout.addWidget(self.checkbox_view_visited, 5, 1)

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
            contrast_limits = [0, 1]
        )

        # Labels layer (hide it by default)
        self.labels_layer = self.viewer.add_labels(
            self.current_labels_chunk(),
            visible=False
        )

        # Points layer
        self.points_layer = self.viewer.add_points(
            self.current_visible_nodes(),
            face_color='red',
            size=1,
        )
        self.points_layer.events.data.connect(self._point_move_update_path)

        # Shapes layer (path)
        self.shapes_layer = self.viewer.add_shapes(
            self.current_visible_nodes(),
            shape_type='path',
            edge_color='red',
            edge_width=0.2
        )

        # Shapes layer (path) in the minimap viewer
        self.minimap_path_layer = self.minimap_viewer.add_shapes(
            self.current_visited_locs(),
            shape_type='path',
            edge_color='red',
            edge_width=1
        )

        # Shapes layer (bounding box) in the minimap viewer
        self.minimap_shapes_layer = self.minimap_viewer.add_shapes(
            self.current_bbox(),
            shape_type='rectangle',
            edge_color='red',
            edge_width=5,
            face_color='transparent',
            name="Current location"
        )
        self.minimap_shapes_layer.mode = 'SELECT'

        # Moving the bounding box updates the 3D view
        self.minimap_shapes_layer.events.set_data.connect(self._test)

        self._update_view()

    def _test(self, e):
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
        # self._update_minimap_bbox()
        self._update_overlay()

    def _update_image(self):
        self.image_layer.data = self.current_image_chunk()

    def _update_labels(self):
        self.labels_layer.data = self.current_labels_chunk()

    def _update_shapes(self):
        visible_nodes = self.current_visible_nodes()
        if len(visible_nodes) < 2:  # Quick hack - hide the layer.
            self.shapes_layer.visible = False
            return
        else:
            self.shapes_layer.visible = True
            
        self.shapes_layer.data = self.current_visible_nodes()
    
    def _update_points(self):
        visible_nodes = self.current_visible_nodes()
        if len(visible_nodes) < 2:  # Quick hack - hide the layer.
            self.points_layer.visible = False
            return
        else:
            self.points_layer.visible = True
            
        self.points_layer.data = self.current_visible_nodes()

    def _update_minimap_path(self):
        if self.checkbox_view_visited.isChecked():
            self.minimap_path_layer.data = self.current_visited_locs()
        else:
            self.minimap_path_layer.data = self.all_node_locs()

    def _update_minimap_bbox(self):
        self.minimap_shapes_layer.data = self.current_bbox()
    
    def _update_overlay(self):
        self.viewer.text_overlay.text = f"Frame {self.current_idx+1} / {self.num_locs}"

    def current_visited_locs(self):
        return self.path_coordinates[:(max(self.current_idx, 2)), 1:]
    
    def all_node_locs(self):
        return self.path_coordinates[:, 1:]

    def current_bbox(self):
        return get_bbox_location(
            self.img,
            center_loc=self.center_loc,
            chunk_shape=self.chunk_shape
        )
    
    def current_visible_nodes(self):
        return get_visible_nodes(
            self.img,
            center_loc=self.center_loc,
            chunk_shape=self.chunk_shape,
            path_coordinates=self.path_coordinates
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
            self.current_idx += 1
            self.center_loc = self.path_coordinates[self.current_idx]
        else:
            return

        self._update_view()

    def move_backward(self, *args, **kwargs):
        if self.current_idx - 1 >= 0:
            self.current_idx -= 1
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

    def _point_move_update_path(self, event):

        # TODO update path_coordinates
        global_idx = get_visible_nodes_idx(
            self.img,
            center_loc=self.center_loc,
            chunk_shape=self.chunk_shape,
            path_coordinates=self.path_coordinates
        )
        # first index that is True
        global_shift = np.where(global_idx)[0][0]
        
        if event.action == "changed": 

            edited_point_idx = event.data_indices[0]
            new_location = event.value[edited_point_idx] + self.center_loc - np.asarray(self.chunk_shape) // 2

            # update path_coordinates
            if np.any(new_location != self.path_coordinates[global_shift + edited_point_idx]):
                self.path_coordinates[global_shift + edited_point_idx] = new_location

            # update path
            self.shapes_layer.data = event.value

        elif event.action == "removed":
            self.shapes_layer.data = event.value

            # delete from numpy array
            self.path_coordinates = np.delete(self.path_coordinates, global_shift + event.data_indices, axis=0)

"""
Launch viewer
"""
minimap_viewer = napari.view_image(da.max(img, axis=0).compute(), contrast_limits=[0, 1], multiscale=False)

viewer = napari.Viewer(ndisplay=3)

skeleton_walker = NeuronSkeletonWalker(img, seg, path_coordinates, viewer, minimap_viewer)

# callback to update the points layer
viewer.window.add_dock_widget(skeleton_walker, name="Neuron walker")


napari.run()

