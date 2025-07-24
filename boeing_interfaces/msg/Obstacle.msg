# Obstacle.msg

# 3D Position (x,y,z coordinates)
geometry_msgs/Point position

# Bounding Box Dimensions (width, height, length)
float32 width
float32 height
float32 length

# Precomputed volume (optional)
float32 volume

# List of surface normal vectors (3D vectors)
geometry_msgs/Vector3[] surface_normals

# Shape category lower value means more ellipsoid
float32 shape

# Outcrop flag (boolean)
bool outcrop_flag
