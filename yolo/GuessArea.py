import math


def side_size_from_time(alpha, beta, t, max_dt=1.0):
    if t < 0:
        return alpha
    if t > max_dt:
        return alpha * math.exp(beta)
    return alpha * math.exp(beta * t / max_dt)


def guess_area(x, y, object_abs_max_side_size, dt_since_last_detection, image_side_size=1):
    alpha = object_abs_max_side_size
    beta = math.log(image_side_size / alpha)
    side_size_for_dt = side_size_from_time(alpha, beta, dt_since_last_detection, 5)

    left = x - side_size_for_dt / 2
    right = x + side_size_for_dt / 2
    if left < 0:
        right -= left
        left = 0
    if right > 1.0:
        left -= right - 1
        right = 1

    top = y - side_size_for_dt / 2
    bottom = y + side_size_for_dt / 2
    if top < 0:
        bottom -= top
        top = 0
    if bottom > 1.0:
        top -= bottom - 1
        bottom = 1

    return left, top, right, bottom
