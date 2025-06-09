INPUT_SCHEMA = {
    "prompt": {
        "datatype": "STRING",
        "required": True,
        "shape": [1],
        "example": ["A futuristic city with flying cars and neon lights at night"]
    },
    "negative_prompt": {
        "datatype": "STRING",
        "required": False,
        "shape": [1],
        "example": ["blurry, low quality, distorted, static"]
    },
    "height": {
        "datatype": "INT32",
        "required": False,
        "shape": [1],
        "example": [480]
    },
    "width": {
        "datatype": "INT32", 
        "required": False,
        "shape": [1],
        "example": [480]
    },
    "num_frames": {
        "datatype": "INT32",
        "required": False,
        "shape": [1],
        "example": [81]
    }
}
