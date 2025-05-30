import numpy as np

import maya.OpenMaya as om
import maya.cmds as mc


def barycentric_coordinates(
    point1: om.MPoint, point2: om.MPoint, point3: om.MPoint, target_point: om.MPoint
) -> tuple[float, float, float]:
    """
    Calculate barycentric coordinates of target_point with respect to triangle (point1, point2, point3).

    Parameters:
        point1, point2, point3, target_point: (3,) numpy arrays or list-like — 3D coordinates.

    Returns:
        A tuple of 3 floats: barycentric weights (w1, w2, w3) corresponding to point1, point2, point3.
    """
    point1 = [point1.x, point1.y, point1.z]
    point2 = [point2.x, point2.y, point2.z]
    point3 = [point3.x, point3.y, point3.z]
    target_point = [target_point.x, target_point.y, target_point.z]

    p1 = np.array(point1, dtype=float)
    p2 = np.array(point2, dtype=float)
    p3 = np.array(point3, dtype=float)
    p = np.array(target_point, dtype=float)

    v0 = p2 - p1
    v1 = p3 - p1
    v2 = p - p1

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if denom == 0:
        raise ValueError("The triangle is degenerate (area is zero).")

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return u, v, w


def main():
    locator_name, geom_name = mc.ls(sl=True)

    sel_list = om.MSelectionList()
    sel_list.add(geom_name)

    geom_dag = om.MDagPath()
    sel_list.getDagPath(0, geom_dag)
    geom_dag.extendToShape()

    mesh_intersector = om.MMeshIntersector()
    mesh_intersector.create(geom_dag.node(), geom_dag.inclusiveMatrix())

    locator_position = mc.xform(locator_name, q=True, t=True, ws=True)
    locator_point = om.MPoint(*locator_position)

    point_on_mesh = om.MPointOnMesh()
    mesh_intersector.getClosestPoint(locator_point, point_on_mesh)
    face_id = point_on_mesh.faceIndex()
    tri_id = point_on_mesh.triangleIndex()

    dummy_util = om.MScriptUtil()
    dummy_int_ptr = dummy_util.asIntPtr()
    current_face = om.MItMeshPolygon(geom_dag)
    point_array = om.MPointArray()
    vtx_ids = om.MIntArray()
    current_face.setIndex(face_id, dummy_int_ptr)
    current_face.getTriangle(tri_id, point_array, vtx_ids, om.MSpace.kWorld)

    point: om.MPoint = point_on_mesh.getPoint()

    barycentric_weights = barycentric_coordinates(
        point_array[0], point_array[1], point_array[2], point
    )

    skin_name = mc.ls(mc.listHistory(geom_name), type="skinCluster")[0]

    weights0 = np.array(mc.getAttr(f"{skin_name}.wl[{vtx_ids[0]}].w[*]"))
    weights1 = np.array(mc.getAttr(f"{skin_name}.wl[{vtx_ids[1]}].w[*]"))
    weights2 = np.array(mc.getAttr(f"{skin_name}.wl[{vtx_ids[2]}].w[*]"))

    weighted_weights = (
        (weights0 * barycentric_weights[0])
        + (weights1 * barycentric_weights[1])
        + (weights2 * barycentric_weights[2])
    )

    inf_names = mc.skinCluster(skin_name, q=True, inf=True)
    current_infs = []
    current_weights = []
    for i, weight in enumerate(weighted_weights):
        if not weight:
            continue
        current_weights.append(weight)
        current_infs.append(inf_names[i])

    output_locator = mc.spaceLocator()[0]
    mc.xform(output_locator, t=[point.x, point.y, point.z], ws=True)
    parcons = mc.parentConstraint(current_infs + [output_locator], mo=True)[0]

    for i, weight in enumerate(current_weights):
        mc.setAttr(f"{parcons}.w{i}", weight)
