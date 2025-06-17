import cv2
import numpy as np
from shot_detection import ShotDetector

def visualize_shots(video_path: str, shot_frames: list):
    """
    Visualiza os frames onde arremessos foram detectados.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Verificar se este frame contém um arremesso
        for shot_frame, ball_bbox in shot_frames:
            if frame_count == shot_frame:
                # Desenhar bounding box da bola
                x, y, w, h = ball_bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "ARREMESSO!", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Mostrar frame
        cv2.imshow('Detecção de Arremesso', frame)
        
        # Sair com 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Inicializar detector
    detector = ShotDetector()
    
    # Processar vídeo
    video_path = "app/videos/user.mp4"  # Ajuste para o caminho do seu vídeo
    shot_frames = detector.process_video(video_path)
    
    # Mostrar resultados
    print(f"Detectados {len(shot_frames)} arremessos!")
    for frame_num, bbox in shot_frames:
        print(f"Frame {frame_num}: Bola detectada em {bbox}")
    
    # Visualizar resultados
    visualize_shots(video_path, shot_frames)

if __name__ == "__main__":
    main() 