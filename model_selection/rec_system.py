def alphanum_to_int(string):
    """encodes alphanumeric string to int"""
    import math
    return int.from_bytes(string.encode(), 'little')

def int_to_alphanum(string):
    """decodes int to alphaumeric string"""
    import math
    return n.to_bytes(math.ceil(n.bit_length() / 8), 'little').decode()