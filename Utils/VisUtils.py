import math

def time_convert(sec):
    mins = math.floor(sec // 60)
    sec = sec % 60
    hours = math.floor(mins // 60)
    mins = mins % 60
    return f"{hours}h, {mins}m, {sec:.2f}s"