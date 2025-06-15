# app/core/constants.py

IMG_SIZE = (480, 640)  # Formato padrão da imagem do modelo de pose
POINT_COLOR = (0, 255, 0)
LINE_COLOR = (255, 0, 0)
RADIUS = 4
THICKNESS = 2

POSE_CONNECTIONS = [
    (11, 13), (13, 15),  # Braço esquerdo
    (12, 14), (14, 16),  # Braço direito
    (11, 12),            # Ombros
    (11, 23), (12, 24),  # Tronco para quadril
    (23, 24),            # Quadril
    (23, 25), (25, 27),  # Perna esquerda
    (24, 26), (26, 28),  # Perna direita
]
