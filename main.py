import cv2
import time
import os
import requests
from collections import deque
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION — modifie ces valeurs dans .env
# ─────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Modèle à utiliser :
# - "yolov8n.pt"  → modèle de base COCO (détecte cell phone nativement)
# - "best_v1.pt" / "best_v2.pt" → modèles fine-tunés
MODEL_PATH = "best_v2.pt"

# Classe cible — laisser None pour auto-détecter depuis le modèle
TARGET_CLASS = None

# Fenêtre glissante : sur WINDOW secondes, si détecté >= DETECTION_DURATION → alerte
# (résistant aux coupures brèves de détection)
WINDOW = 5.0             # durée de la fenêtre glissante (secondes)
DETECTION_DURATION = 3.0 # secondes de détection requises dans la fenêtre

# Cooldown entre deux messages Telegram (en secondes) — évite le spam
COOLDOWN = 30

# ─────────────────────────────────────────────
# FONCTION TELEGRAM
# ─────────────────────────────────────────────
def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"[Telegram] Message envoyé : {message}")
        else:
            print(f"[Telegram] Erreur : {response.text}")
    except Exception as e:
        print(f"[Telegram] Connexion échouée : {e}")


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
def main():
    print(f"[Init] Chargement du modèle : {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    available_classes = list(model.names.values())
    print(f"[Init] Classes disponibles dans ce modèle : {available_classes}")

    target = TARGET_CLASS
    if target is None:
        for candidate in ["cell phone", "phone", "cellphone"]:
            if candidate in available_classes:
                target = candidate
                break
        if target is None:
            target = available_classes[0]
        print(f"[Init] Classe cible auto-détectée : '{target}'")
    elif target not in available_classes:
        print(f"[Attention] Classe '{target}' absente du modèle. Classes dispo : {available_classes}")
        target = available_classes[0]
    else:
        print(f"[Init] Classe cible : '{target}'")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Erreur] Impossible d'ouvrir la webcam.")
        return

    print("[Init] Webcam ouverte. Appuie sur 'q' pour quitter.")

    # Historique des frames : (timestamp, detected) sur les WINDOW dernières secondes
    history = deque()
    last_alert_time = 0
    alert_sent = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Erreur] Frame non lue.")
            break

        # ── Détection YOLOv8 ──────────────────
        results = model(frame, verbose=False)
        detected = False

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                if class_name == target and confidence > 0.5:
                    detected = True

        # ── Fenêtre glissante ─────────────────
        now = time.time()
        history.append((now, detected))

        # Supprime les entrées hors de la fenêtre
        while history and now - history[0][0] > WINDOW:
            history.popleft()

        # Calcule le temps détecté dans la fenêtre en sommant les intervalles
        detected_time = 0.0
        for i in range(1, len(history)):
            if history[i - 1][1]:  # si la frame précédente était détectée
                detected_time += history[i][0] - history[i - 1][0]

        # ── Logique de déclenchement ──────────
        if detected_time >= DETECTION_DURATION:
            print(f"[Détection] '{target}' : {detected_time:.1f}s / {WINDOW}s  ", end="\r")
            if not alert_sent and now - last_alert_time > COOLDOWN:
                send_telegram(f"Alerte : '{target}' détecté {detected_time:.1f}s sur les {WINDOW}s !")
                last_alert_time = now
                alert_sent = True
        else:
            if detected_time > 0:
                print(f"[Suivi]     '{target}' : {detected_time:.1f}s / {DETECTION_DURATION}s requis  ", end="\r")
            alert_sent = False

        # ── Affichage ─────────────────────────
        annotated = results[0].plot()
        status = "DETECTE" if detected else "En attente..."
        color = (0, 0, 255) if detected else (0, 255, 0)
        cv2.putText(annotated, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        # Affiche le compteur de temps détecté sur le feed vidéo
        cv2.putText(annotated, f"{detected_time:.1f}s / {DETECTION_DURATION}s", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("DCP — Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[Fin] Programme arrêté.")


if __name__ == "__main__":
    main()
