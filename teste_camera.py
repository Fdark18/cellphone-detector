#!/usr/bin/env python3
"""
Sistema de Detecção de Uso de Celular - Versão Linux 
================================================================

Monitoramento em tempo real com máxima performance.
Otimizado para RTX 2050 4GB em ambiente Linux.

Uso:
    python3 teste_camera.py

Autor: Jhonatan Novais
finalidade: Sistema de Monitoramento Industrial
Versão: 2.0 (Linux Optimized)
"""

import cv2
from ultralytics import YOLO
from datetime import datetime
import os
import sys
import signal
import threading
import queue


class DetectorCelularLinux:
    """
    Detector otimizado para Linux com threading e performance máxima.
    """
    
    def __init__(self, modelo_path, conf_threshold=0.6):
        """
        Inicializa detector com configurações otimizadas.
        
        Args:
            modelo_path: Caminho para modelo .pt
            conf_threshold: Threshold de confiança (0.0-1.0)
        """
        if not os.path.exists(modelo_path):
            raise FileNotFoundError(f"Modelo não encontrado: {modelo_path}")
        
        self.model = YOLO(modelo_path)
        self.model.fuse()  # Otimização de layers
        
        self.conf_threshold = conf_threshold
        self.usando_celular = False
        self.tempo_inicio = None
        
        # Diretórios
        self.dir_deteccoes = 'deteccoes'
        os.makedirs(self.dir_deteccoes, exist_ok=True)
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running = True
        
        # Log
        self._inicializar_log()
        
        # Signal handler para Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handler para interrupção graceful."""
        print("\n\nEncerrando sistema...")
        self.running = False
    
    def _inicializar_log(self):
        """Cria arquivo de log."""
        self.log_file = os.path.join(
            self.dir_deteccoes,
            f"log_{datetime.now().strftime('%Y%m%d')}.txt"
        )
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"Sistema iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n")
    
    def _registrar_log(self, mensagem):
        """Registra evento no log."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {mensagem}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        print(log_entry.strip())
    
    def _salvar_evidencia(self, frame):
        """Salva evidência fotográfica."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.dir_deteccoes, f"celular_{timestamp}.jpg")
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return filename
    
    def _inference_thread(self):
        """Thread dedicada para inferência GPU."""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                self.result_queue.put((frame, results))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Erro na inferência: {e}")
    
    def _desenhar_interface(self, frame, deteccoes):
        """Desenha interface sobre o frame."""
        h, w = frame.shape[:2]
        
        # Desenhar detecções
        for box in deteccoes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            
            # Cor baseada em confiança
            if conf > 0.8:
                color = (0, 255, 0)  # Verde
            elif conf > 0.6:
                color = (0, 165, 255)  # Laranja
            else:
                color = (0, 0, 255)  # Vermelho
            
            # Retângulo
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Label
            label = f"Celular: {conf:.2%}"
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (x1, y1 - text_h - 10),
                         (x1 + text_w + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Status de uso
        if self.usando_celular:
            tempo_decorrido = (datetime.now() - self.tempo_inicio).seconds
            status = f"USO DETECTADO: {tempo_decorrido}s"
            cor_status = (0, 0, 255)
        else:
            status = "MONITORANDO"
            cor_status = (0, 255, 0)
        
        # Painel de status
        cv2.rectangle(frame, (10, 10), (500, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (500, 80), cor_status, 3)
        cv2.putText(frame, status, (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor_status, 3)
        
        # FPS e info
        info = f"Threshold: {self.conf_threshold:.2f} | ESC: Sair"
        cv2.putText(frame, info, (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def processar_camera(self, camera_id=0, resolucao=(1280, 720)):
        """
        Processa vídeo da câmera com threading otimizado.
        
        Args:
            camera_id: ID da câmera
            resolucao: Tupla (width, height)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Erro ao acessar câmera {camera_id}")
        
        # Configurar câmera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolucao[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucao[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("="*70)
        print("SISTEMA DE DETECÇÃO INICIADO")
        print("="*70)
        print(f"Câmera:          {camera_id}")
        print(f"Resolução:       {resolucao[0]}x{resolucao[1]}")
        print(f"Threshold:       {self.conf_threshold}")
        print(f"Modelo:          Otimizado para GPU")
        print(f"Diretório:       {self.dir_deteccoes}")
        print(f"\nPressione ESC para sair")
        print("="*70)
        
        self._registrar_log(f"Monitoramento iniciado - Câmera {camera_id}")
        
        # Iniciar thread de inferência
        inference_thread = threading.Thread(target=self._inference_thread, daemon=True)
        inference_thread.start()
        
        frame_count = 0
        
        try:
            while self.running:
                ret, frame = cap.read()
                
                if not ret:
                    print("AVISO: Falha ao capturar frame")
                    break
                
                frame_count += 1
                
                # Adicionar frame à fila de inferência
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                
                # Obter resultados
                try:
                    processed_frame, results = self.result_queue.get_nowait()
                    deteccoes = results[0].boxes
                    
                    deteccao_atual = len(deteccoes) > 0
                    
                    # Início de detecção
                    if deteccao_atual and not self.usando_celular:
                        self.usando_celular = True
                        self.tempo_inicio = datetime.now()
                        
                        arquivo = self._salvar_evidencia(frame)
                        self._registrar_log(f"DETECÇÃO INICIADA - Evidência: {arquivo}")
                    
                    # Fim de detecção
                    elif not deteccao_atual and self.usando_celular:
                        tempo_uso = (datetime.now() - self.tempo_inicio).seconds
                        self._registrar_log(f"DETECÇÃO FINALIZADA - Duração: {tempo_uso}s")
                        
                        self.usando_celular = False
                        self.tempo_inicio = None
                    
                    # Desenhar interface
                    frame = self._desenhar_interface(frame, deteccoes)
                
                except queue.Empty:
                    pass
                
                # Exibir
                cv2.imshow('Sistema de Detecção de Celular', frame)
                
                # Verificar tecla
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
        
        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário")
        
        finally:
            self.running = False
            
            if self.usando_celular:
                tempo_uso = (datetime.now() - self.tempo_inicio).seconds
                self._registrar_log(f"DETECÇÃO INTERROMPIDA - Duração: {tempo_uso}s")
            
            cap.release()
            cv2.destroyAllWindows()
            
            self._registrar_log("Sistema finalizado")
            print("\nSistema encerrado")


def main():
    """Função principal."""
    # Configurações
    MODELO_PATH = 'best.pt'
    CAMERA_ID = 0
    CONFIDENCE = 0.6
    RESOLUCAO = (1280, 720)
    
    if not os.path.exists(MODELO_PATH):
        print(f"ERRO: Modelo não encontrado: {MODELO_PATH}")
        print("Execute o treinamento primeiro no notebook")
        sys.exit(1)
    
    try:
        detector = DetectorCelularLinux(
            modelo_path=MODELO_PATH,
            conf_threshold=CONFIDENCE
        )
        
        detector.processar_camera(
            camera_id=CAMERA_ID,
            resolucao=RESOLUCAO
        )
    
    except Exception as e:
        print(f"\nERRO: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()