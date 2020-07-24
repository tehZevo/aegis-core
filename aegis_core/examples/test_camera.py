from aegis_core.camera_node import CameraNode
from aegis_core.imshow_node import ImshowNode
from aegis_core.utils import start_node, start_nodes

start_nodes([
  CameraNode(12400, resize=64),
  ImshowNode(12401, input_url="12400")
])
