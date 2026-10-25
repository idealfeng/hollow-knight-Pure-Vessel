import math

# NOTE ON BOX FORMAT
# Every box in this project is (min_x, min_y, max_x, max_y) - two opposite
# CORNERS, not (x, y, width, height). connection.cs writes them straight from
# Unity's Bounds.min/.max. Treating them as x/y/w/h silently produces
# plausible-looking but wrong geometry, since min_x + max_x/2 happens to be a
# valid-looking number.


def box_center(box):
    """Midpoint of a (min_x, min_y, max_x, max_y) box.

    Single source of truth for this - centre maths was previously duplicated
    (and got the format wrong in one copy), so everything that needs a centre
    goes through here now."""
    min_x, min_y, max_x, max_y = box
    return (min_x + max_x) / 2.0, (min_y + max_y) / 2.0


NO_HAZARD_DISTANCE = 100.0
def nearest_hazard_distance(knight_box, hazard_boxes):
    if not hazard_boxes:
        return NO_HAZARD_DISTANCE
    return min(rectangle_center_distance(knight_box, h) for h in hazard_boxes)


# Offsets are clipped to this. A hazard that spawns across the room would
# otherwise hand the network a huge value that obs_norm has to absorb.
MAX_OFFSET = 50.0


def relative_offsets(knight_box, hazard_boxes, count=2):
    kx, ky = box_center(knight_box)
    out = []
    for i in range(count):
        if i < len(hazard_boxes):
            hx, hy = box_center(hazard_boxes[i])
            out.append(max(-MAX_OFFSET, min(MAX_OFFSET, hx - kx)))
            out.append(max(-MAX_OFFSET, min(MAX_OFFSET, hy - ky)))
        else:
            out.extend((0.0, 0.0))
    return out


def box_distance(box1, box2):
    """Gap between two boxes' edges - 0 when they overlap."""
    min_x1, min_y1, max_x1, max_y1 = box1
    min_x2, min_y2, max_x2, max_y2 = box2

    dx = max(min_x1 - max_x2, min_x2 - max_x1, 0)
    dy = max(min_y1 - max_y2, min_y2 - max_y1, 0)

    return math.hypot(dx, dy)


def boxes_partial_intersect(box1, box2):
    """True when the boxes overlap but neither fully contains the other."""
    min_x1, min_y1, max_x1, max_y1 = box1
    min_x2, min_y2, max_x2, max_y2 = box2

    intersect = (
        min_x1 < max_x2 and
        max_x1 > min_x2 and
        min_y1 < max_y2 and
        max_y1 > min_y2
    )

    if not intersect:
        return False

    box1_inside_box2 = (
        min_x1 >= min_x2 and
        min_y1 >= min_y2 and
        max_x1 <= max_x2 and
        max_y1 <= max_y2
    )

    box2_inside_box1 = (
        min_x2 >= min_x1 and
        min_y2 >= min_y1 and
        max_x2 <= max_x1 and
        max_y2 <= max_y1
    )

    return not (box1_inside_box2 or box2_inside_box1)




def calculate_area(coordinates):
    x, y, x2, y2 = coordinates
    return abs(x-x2) * abs(y-y2)



def rectangle_center_distance(rect1, rect2):
    # Centres come from box_center() so the corner-vs-width/height mix-up
    # that used to live here cannot come back in one copy only.
    center1_x, center1_y = box_center(rect1)
    center2_x, center2_y = box_center(rect2)

    return math.hypot(center2_x - center1_x, center2_y - center1_y)
