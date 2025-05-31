import numpy as np
import maya.OpenMaya as om
import maya.cmds as mc


def barycentric_coordinates(
    point1: om.MPoint, point2: om.MPoint, point3: om.MPoint, target_point: om.MPoint
) -> tuple[float, float, float]:
    """
    Computes the barycentric coordinates of `target_point` relative to the triangle defined by
    `point1`, `point2`, and `point3`.

    Barycentric coordinates express a point's location as a weighted sum of a triangle's vertices.

    Parameters:
        point1, point2, point3 : om.MPoint
            The three vertices of the reference triangle.
        target_point : om.MPoint
            The point whose barycentric coordinates are to be determined.

    Returns:
        tuple[float, float, float] : (w1, w2, w3)
            Barycentric weights corresponding to `point1`, `point2`, and `point3`.
    """
    # Convert MPoints to numpy arrays for numerical operations
    p1 = np.array([point1.x, point1.y, point1.z], dtype=float)
    p2 = np.array([point2.x, point2.y, point2.z], dtype=float)
    p3 = np.array([point3.x, point3.y, point3.z], dtype=float)
    p = np.array([target_point.x, target_point.y, target_point.z], dtype=float)

    # Compute edge vectors relative to the first triangle vertex
    v0 = p2 - p1  # Vector from point1 to point2
    v1 = p3 - p1  # Vector from point1 to point3
    v2 = p - p1  # Vector from point1 to target_point

    # Compute dot products for use in the barycentric coordinate calculation
    d00 = np.dot(v0, v0)  # Squared length of v0
    d01 = np.dot(v0, v1)  # Dot product of v0 and v1
    d11 = np.dot(v1, v1)  # Squared length of v1
    d20 = np.dot(v2, v0)  # Dot product of v2 and v0
    d21 = np.dot(v2, v1)  # Dot product of v2 and v1

    # Compute barycentric weights using determinant method
    denom = d00 * d11 - d01 * d01  # Determinant of the Gram matrix
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return u, v, w


def main():
    """
    Determines the skinning weights for a given locator based on its closest mesh intersection.
    Uses barycentric interpolation to blend weights of adjacent triangle vertices.
    """
    # Get selected objects: locator and geometry
    locator_name, geom_name = mc.ls(sl=True)

    # Obtain the mesh shape as an MDagPath
    sel_list = om.MSelectionList()
    sel_list.add(geom_name)
    geom_dag = om.MDagPath()
    sel_list.getDagPath(0, geom_dag)
    geom_dag.extendToShape()

    # Get the world-space position of the locator as an MPoint
    locator_position = mc.xform(locator_name, q=True, t=True, ws=True)
    locator_point = om.MPoint(*locator_position)

    # Initialize mesh intersector and compute closest point on the mesh
    mesh_intersector = om.MMeshIntersector()
    mesh_intersector.create(geom_dag.node(), geom_dag.inclusiveMatrix())

    point_on_mesh = om.MPointOnMesh()
    mesh_intersector.getClosestPoint(locator_point, point_on_mesh)

    # Retrieve the intersection's face and triangle indices
    face_id = point_on_mesh.faceIndex()
    tri_id = point_on_mesh.triangleIndex()

    # Initialize mesh iterator and extract triangle vertex IDs and positions
    dummy_util = om.MScriptUtil()
    dummy_int_ptr = dummy_util.asIntPtr()
    current_face = om.MItMeshPolygon(geom_dag)
    point_array = om.MPointArray()
    vtx_ids = om.MIntArray()
    current_face.setIndex(face_id, dummy_int_ptr)
    current_face.getTriangle(tri_id, point_array, vtx_ids, om.MSpace.kWorld)

    # Get the exact intersection position
    point = point_on_mesh.getPoint()

    # Compute barycentric weights for interpolation
    barycentric_weights = barycentric_coordinates(
        point_array[0], point_array[1], point_array[2], point
    )

    # Retrieve skin cluster affecting the geometry
    skin_name = mc.ls(mc.listHistory(geom_name), type="skinCluster")[0]

    # Extract vertex skin weights
    weights0 = np.array(mc.getAttr(f"{skin_name}.wl[{vtx_ids[0]}].w[*]"))
    weights1 = np.array(mc.getAttr(f"{skin_name}.wl[{vtx_ids[1]}].w[*]"))
    weights2 = np.array(mc.getAttr(f"{skin_name}.wl[{vtx_ids[2]}].w[*]"))

    # Compute weighted skin influences using barycentric interpolation
    weighted_weights = (
        (weights0 * barycentric_weights[0])
        + (weights1 * barycentric_weights[1])
        + (weights2 * barycentric_weights[2])
    )

    # Retrieve influence names and filter non-zero weights
    inf_names = mc.skinCluster(skin_name, q=True, inf=True)
    current_infs = [inf_names[i] for i, weight in enumerate(weighted_weights) if weight]
    current_weights = [weight for weight in weighted_weights if weight]

    # Create output locator at interpolated position
    output_locator = mc.spaceLocator()[0]
    mc.xform(output_locator, t=[point.x, point.y, point.z], ws=True)

    # Apply parent constraint to maintain influence blending
    parcons = mc.parentConstraint(current_infs + [output_locator], mo=True)[0]

    # Set constraint weight values
    for i, weight in enumerate(current_weights):
        mc.setAttr(f"{parcons}.w{i}", weight)
