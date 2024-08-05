"""
evaluation scripts, developed by Mask3d
https://github.com/JonasSchult/Mask3D
"""
import os, sys
import csv

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)
try:
    import imageio
except:
    print("Please install the module 'imageio' for image processing, e.g.")
    print("pip install imageio")
    sys.exit(-1)

# print an error message and quit
def print_error(message, user_fault=False):
    sys.stderr.write("ERROR: " + str(message) + "\n")
    if user_fault:
        sys.exit(2)
    sys.exit(-1)


# if string s represents an int
def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(
    filename, label_from="raw_category", label_to="nyu40id"
):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    # if ints convert
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping


# input: scene_types.txt or scene_types_all.txt
def read_scene_types_mapping(filename, remove_spaces=True):
    assert os.path.isfile(filename)
    mapping = dict()
    lines = open(filename).read().splitlines()
    lines = [line.split("\t") for line in lines]
    if remove_spaces:
        mapping = {x[1].strip(): int(x[0]) for x in lines}
    else:
        mapping = {x[1]: int(x[0]) for x in lines}
    return mapping


# color by label
def visualize_label_image(filename, image):
    height = image.shape[0]
    width = image.shape[1]
    vis_image = np.zeros([height, width, 3], dtype=np.uint8)
    color_palette = create_color_palette()
    for idx, color in enumerate(color_palette):
        vis_image[image == idx] = color
    imageio.imwrite(filename, vis_image)


# color by different instances (mod length of color palette)
def visualize_instance_image(filename, image):
    height = image.shape[0]
    width = image.shape[1]
    vis_image = np.zeros([height, width, 3], dtype=np.uint8)
    color_palette = create_color_palette()
    instances = np.unique(image)
    for idx, inst in enumerate(instances):
        vis_image[image == inst] = color_palette[inst % len(color_palette)]
    imageio.imwrite(filename, vis_image)


# color palette for nyu40 labels
def create_color_palette():
    return [
        (0, 0, 0),
        (174, 199, 232),  # wall
        (152, 223, 138),  # floor
        (31, 119, 180),  # cabinet
        (255, 187, 120),  # bed
        (188, 189, 34),  # chair
        (140, 86, 75),  # sofa
        (255, 152, 150),  # table
        (214, 39, 40),  # door
        (197, 176, 213),  # window
        (148, 103, 189),  # bookshelf
        (196, 156, 148),  # picture
        (23, 190, 207),  # counter
        (178, 76, 76),
        (247, 182, 210),  # desk
        (66, 188, 102),
        (219, 219, 141),  # curtain
        (140, 57, 197),
        (202, 185, 52),
        (51, 176, 203),
        (200, 54, 131),
        (92, 193, 61),
        (78, 71, 183),
        (172, 114, 82),
        (255, 127, 14),  # refrigerator
        (91, 163, 138),
        (153, 98, 156),
        (140, 153, 101),
        (158, 218, 229),  # shower curtain
        (100, 125, 154),
        (178, 127, 135),
        (120, 185, 128),
        (146, 111, 194),
        (44, 160, 44),  # toilet
        (112, 128, 144),  # sink
        (96, 207, 209),
        (227, 119, 194),  # bathtub
        (213, 92, 176),
        (94, 106, 211),
        (82, 84, 163),  # otherfurn
        (100, 85, 144),
    ]
    
    
    
    import os, sys
import json

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)

# matrix: 4x4 np array
# points Nx3 np array
def transform_points(matrix, points):
    assert len(points.shape) == 2 and points.shape[1] == 3
    num_points = points.shape[0]
    p = np.concatenate([points, np.ones((num_points, 1))], axis=1)
    p = np.matmul(matrix, np.transpose(p))
    p = np.transpose(p)
    p[:, :3] /= p[:, 3, None]
    return p[:, :3]


def export_ids(filename, ids):
    with open(filename, "w") as f:
        for id in ids:
            f.write("%d\n" % id)


def load_ids(filename):
    ids = open(filename).read().splitlines()
    ids = np.array(ids, dtype=np.int64)
    return ids


def read_mesh_vertices(filename):
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
    return vertices


# export 3d instance labels for instance evaluation
def export_instance_ids_for_eval(filename, label_ids, instance_ids):
    assert label_ids.shape[0] == instance_ids.shape[0]
    output_mask_path_relative = "pred_mask"
    name = os.path.splitext(os.path.basename(filename))[0]
    output_mask_path = os.path.join(
        os.path.dirname(filename), output_mask_path_relative
    )
    if not os.path.isdir(output_mask_path):
        os.mkdir(output_mask_path)
    insts = np.unique(instance_ids)
    zero_mask = np.zeros(shape=(instance_ids.shape[0]), dtype=np.int32)
    with open(filename, "w") as f:
        for idx, inst_id in enumerate(insts):
            if inst_id == 0:  # 0 -> no instance for this vertex
                continue
            output_mask_file = os.path.join(
                output_mask_path_relative, name + "_" + str(idx) + ".txt"
            )
            loc = np.where(instance_ids == inst_id)
            label_id = label_ids[loc[0][0]]
            f.write("%s %d %f\n" % (output_mask_file, label_id, 1.0))
            # write mask
            mask = np.copy(zero_mask)
            mask[loc[0]] = 1
            export_ids(output_mask_file, mask)


# ------------ Instance Utils ------------ #


class Instance(object):
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, mesh_vert_instances, instance_id):
        if instance_id == -1:
            return
        self.instance_id = int(instance_id)
        self.label_id = int(self.get_label_id(instance_id))
        self.vert_count = int(
            self.get_instance_verts(mesh_vert_instances, instance_id)
        )
        self.mask = self.get_instance_mask(mesh_vert_instances, instance_id)

    def get_label_id(self, instance_id):
        return int(instance_id // 1000)

    def get_instance_verts(self, mesh_vert_instances, instance_id):
        return (mesh_vert_instances == instance_id).sum()
    
    def get_instance_mask(self, mesh_vert_instances, instance_id):
        return (mesh_vert_instances == instance_id)


    def to_json(self):
        return json.dumps(
            self, default=lambda o: o.__dict__, sort_keys=True, indent=4
        )

    def to_dict(self):
        dict = {}
        dict["instance_id"] = self.instance_id
        dict["label_id"] = self.label_id
        dict["vert_count"] = self.vert_count
        dict["med_dist"] = self.med_dist
        dict["dist_conf"] = self.dist_conf
        dict["mask"] = self.mask
        return dict

    def from_json(self, data):
        self.instance_id = int(data["instance_id"])
        self.label_id = int(data["label_id"])
        self.vert_count = int(data["vert_count"])
        if "med_dist" in data:
            self.med_dist = float(data["med_dist"])
            self.dist_conf = float(data["dist_conf"])

    def __str__(self):
        return "(" + str(self.instance_id) + ")"


def read_instance_prediction_file(filename, pred_path):
    lines = open(filename).read().splitlines()
    instance_info = {}
    abs_pred_path = os.path.abspath(pred_path)
    for line in lines:
        parts = line.split(" ")
        if len(parts) != 3:
            util.print_error(
                "invalid instance prediction file. Expected (per line): [rel path prediction] [label id prediction] [confidence prediction]"
            )
        if os.path.isabs(parts[0]):
            util.print_error(
                "invalid instance prediction file. First entry in line must be a relative path"
            )
        mask_file = os.path.join(os.path.dirname(filename), parts[0])
        mask_file = os.path.abspath(mask_file)
        # check that mask_file lives inside prediction path
        if os.path.commonprefix([mask_file, abs_pred_path]) != abs_pred_path:
            util.print_error(
                "predicted mask {} in prediction text file {} points outside of prediction path.".format(
                    mask_file, filename
                )
            )

        info = {}
        info["label_id"] = int(float(parts[1]))
        info["conf"] = float(parts[2])
        instance_info[mask_file] = info
    return instance_info


def get_instances(ids, class_ids, class_labels, id2label):
    instances = {}
    for label in class_labels:
        instances[label] = []
    instance_ids = np.unique(ids)
    for id in instance_ids:
        if id == 0:
            continue
        inst = Instance(ids, id)
        if inst.label_id in class_ids:
            instances[id2label[inst.label_id]].append(inst.to_dict())
    return instances