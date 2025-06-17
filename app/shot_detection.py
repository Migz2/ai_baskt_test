import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, List, Optional, Any

class ShotDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Parâmetros para detecção de movimento
        self.velocity_threshold = 0.05  # Limiar mais sensível para detectar movimento súbito
        self.head_height_threshold = 0.8  # Altura relativa da cabeça para referência
        
        # Parâmetros para detecção da bola
        self.ball_color_lower = np.array([10, 70, 70])  # HSV - mais abrangente
        self.ball_color_upper = np.array([40, 255, 255])
        self.min_ball_area = 50  # Permitir bolas menores
        
    def detect_arm_movement(self, landmarks: Any) -> bool:
        """
        Detecta movimento súbito do braço direito acima da cabeça.
        """
        if not landmarks:
            return False
            
        # Obter keypoints relevantes
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        
        # Verificar se o punho está acima da cabeça
        wrist_above_head = right_wrist.y < nose.y
        
        # Calcular velocidade do cotovelo e punho
        elbow_velocity = np.sqrt(
            (right_elbow.x - right_shoulder.x)**2 + 
            (right_elbow.y - right_shoulder.y)**2
        )
        
        wrist_velocity = np.sqrt(
            (right_wrist.x - right_elbow.x)**2 + 
            (right_wrist.y - right_elbow.y)**2
        )
        
        # Verificar movimento súbito
        sudden_movement = (elbow_velocity > self.velocity_threshold or 
                         wrist_velocity > self.velocity_threshold)
        
        return wrist_above_head and sudden_movement
    
    def detect_ball(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detecta a bola usando detecção de cor em HSV.
        Retorna a bounding box da bola (x, y, w, h) ou None se não encontrada.
        """
        # Converter para HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Criar máscara para a cor da bola
        mask = cv2.inRange(hsv, self.ball_color_lower, self.ball_color_upper)
        
        # Aplicar operações morfológicas para remover ruído
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por área
        ball_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_ball_area]
        
        if ball_contours:
            # Pegar o maior contorno
            largest_contour = max(ball_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return (x, y, w, h)
        
        return None
    
    def detect_shot(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """
        Detecta um arremesso combinando detecção de movimento do braço e da bola.
        Retorna (shot_detected, ball_bbox)
        """
        # Converter frame para RGB para o MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar pose
        results = self.pose.process(frame_rgb)
        
        # Detectar movimento do braço
        arm_movement = self.detect_arm_movement(results.pose_landmarks)
        
        # Detectar bola
        ball_bbox = self.detect_ball(frame)
        
        # Um arremesso é detectado quando há movimento do braço e a bola é detectada
        shot_detected = arm_movement and ball_bbox is not None
        
        return shot_detected, ball_bbox
    
    def process_video(self, video_path: str) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        """
        Processa um vídeo e retorna uma lista de frames onde arremessos foram detectados
        junto com as bounding boxes da bola. Evita múltiplas detecções consecutivas do mesmo arremesso.
        """
        cap = cv2.VideoCapture(video_path)
        shot_frames = []
        frame_count = 0
        shot_active = False  # Flag para saber se já estamos em um arremesso
        min_frame_gap = 30  # Mínimo de frames entre arremessos (ajustável, ex: 1 segundo a 30fps)
        last_shot_frame = -min_frame_gap  # Inicializa para permitir detecção no início

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            shot_detected, ball_bbox = self.detect_shot(frame)

            if shot_detected and not shot_active and (frame_count - last_shot_frame >= min_frame_gap):
                shot_frames.append((frame_count, ball_bbox))
                shot_active = True  # Marca que já detectou o arremesso
                last_shot_frame = frame_count

            if not shot_detected:
                shot_active = False  # Reseta a flag quando o arremesso termina

            frame_count += 1

        cap.release()
        return shot_frames
    
    def __del__(self):
        self.pose.close() 