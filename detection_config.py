# Detection configuration
DETECTION_CONFIG = {
    'face_recognition': {
        'enabled': True,
        'tolerance': 0.6,      # Lower = more strict matching
        'scan_every_n_frames': 3,  # Process every Nth frame for better performance
        'min_face_size': 20    # Minimum face size in pixels
    },
    'mask_detection': {
        'enabled': True,
        'confidence_threshold': 0.7
    },
    'weapon_detection': {
        'enabled': True,
        'confidence_threshold': 0.5,
        'classes': ['gun', 'knife']
    },
    'motion_detection': {
        'enabled': True,
        'sensitivity': 20,     # Higher = more sensitive
        'blur_size': 21,       # Blur size for noise reduction
        'min_area': 500        # Minimum motion area in pixels
    }
}