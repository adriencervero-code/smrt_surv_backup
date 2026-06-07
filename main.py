import cv2
import math
import time
import requests
from collections import deque
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
N8N_WEBHOOK = "http://localhost:5678/webhook/detection"

MODEL_PHONE  = "best_v1.pt"   # détecte les téléphones
MODEL_PERSON = "yolov8n.pt"   # détecte les personnes (classe "person" uniquement)

TARGET_PHONE_CLASS = None     # None = auto-détection depuis MODEL_PHONE

PROXIMITY_THRESHOLD = 200     # distance max (px) entre centres phone et person

CONF_DISPLAY = 0.3            # seuil minimal pour afficher une bbox
CONF_TRIGGER = 0.5            # seuil pour la logique de déclenchement

WINDOW             = 5.0      # durée de la fenêtre glissante (secondes)
DETECTION_DURATION = 3.0      # secondes de détection requises dans la fenêtre
COOLDOWN           = 30       # secondes entre deux alertes Telegram

# ─────────────────────────────────────────────
# ENVOI VIA N8N
# ─────────────────────────────────────────────
def send_telegram(message: str):
    payload = {
        "message":   message,
        "timestamp": time.strftime("%H:%M:%S"),
        "model":     MODEL_PHONE,
    }
    try:
        response = requests.post(N8N_WEBHOOK, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"[n8n] Données envoyées avec succès")
        else:
            print(f"[n8n] Erreur : {response.status_code}")
    except Exception as e:
        print(f"[n8n] Connexion échouée : {e}")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_centers(results, model, target_class, conf_threshold):
    """Retourne les centres (cx, cy) des bboxes pour target_class avec conf >= conf_threshold."""
    centers = []
    for result in results:
        for box in result.boxes:
            if model.names[int(box.cls[0])] == target_class and float(box.conf[0]) >= conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0]
                centers.append((float((x1 + x2) / 2), float((y1 + y2) / 2)))
    return centers


def draw_boxes(frame, results, model, target_class, color):
    """Dessine les bboxes dont la classe == target_class et conf >= CONF_DISPLAY."""
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            cls  = model.names[int(box.cls[0])]
            if cls == target_class and conf >= CONF_DISPLAY:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{cls} {conf:.2f}", (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def closest_distance(centers_a, centers_b):
    """Distance minimale entre deux listes de points (retourne inf si l'une est vide)."""
    if not centers_a or not centers_b:
        return float("inf")
    return min(
        math.hypot(a[0] - b[0], a[1] - b[1])
        for a in centers_a
        for b in centers_b
    )


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
def main():
    print(f"[Init] Chargement de {MODEL_PHONE} (téléphone) ...")
    model_phone = YOLO(MODEL_PHONE)
    print(f"[Init] Chargement de {MODEL_PERSON} (personne) ...")
    model_person = YOLO(MODEL_PERSON)

    # Auto-détection de la classe téléphone dans MODEL_PHONE
    phone_classes = list(model_phone.names.values())
    phone_target  = TARGET_PHONE_CLASS
    if phone_target is None:
        for candidate in ["cell phone", "phone", "cellphone"]:
            if candidate in phone_classes:
                phone_target = candidate
                break
        if phone_target is None:
            phone_target = phone_classes[0]
    print(f"[Init] Classe téléphone : '{phone_target}' | Classe personne : 'person'")
    print(f"[Init] Proximité max : {PROXIMITY_THRESHOLD}px | Fenêtre : {WINDOW}s | Seuil : {DETECTION_DURATION}s")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Erreur] Impossible d'ouvrir la webcam.")
        return
    print("[Init] Webcam ouverte. Appuie sur 'q' pour quitter.")

    history        = deque()
    last_alert_time = 0
    alert_sent     = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Erreur] Frame non lue.")
            break

        # ── Inférence des deux modèles ────────
        phone_results  = model_phone(frame, verbose=False)
        person_results = model_person(frame, verbose=False)

        # Centres des détections au-dessus du seuil de déclenchement
        phone_centers  = get_centers(phone_results,  model_phone,  phone_target, CONF_TRIGGER)
        person_centers = get_centers(person_results, model_person, "person",     CONF_TRIGGER)

        # Détection = téléphone ET personne proches simultanément
        dist     = closest_distance(phone_centers, person_centers)
        detected = dist < PROXIMITY_THRESHOLD

        # ── Fenêtre glissante ─────────────────
        now = time.time()
        history.append((now, detected))
        while history and now - history[0][0] > WINDOW:
            history.popleft()

        detected_time = 0.0
        for i in range(1, len(history)):
            if history[i - 1][1]:
                detected_time += history[i][0] - history[i - 1][0]

        # ── Logique de déclenchement ──────────
        if detected_time >= DETECTION_DURATION:
            print(f"[Alerte]  phone+person proches {detected_time:.1f}s / {WINDOW}s  ", end="\r")
            if not alert_sent and now - last_alert_time > COOLDOWN:
                send_telegram(
                    f"Alerte : téléphone à proximité d'une personne "
                    f"({detected_time:.1f}s sur {WINDOW}s) !"
                )
                last_alert_time = now
                alert_sent = True
        else:
            if detected_time > 0:
                print(f"[Suivi]   {detected_time:.1f}s / {DETECTION_DURATION}s requis  ", end="\r")
            alert_sent = False

        # ── Affichage ─────────────────────────
        annotated = frame.copy()
        draw_boxes(annotated, phone_results,  model_phone,  phone_target, (0, 165, 255))  # orange
        draw_boxes(annotated, person_results, model_person, "person",     (255, 100,  0))  # bleu

        trigger_active = detected_time >= DETECTION_DURATION
        color  = (0, 0, 255) if trigger_active else ((0, 200, 255) if detected else (0, 255, 0))
        status = "ALERTE" if trigger_active else ("PROCHE" if detected else "En attente...")

        cv2.putText(annotated, status,
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(annotated, f"{detected_time:.1f}s / {DETECTION_DURATION}s",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if detected and dist < float("inf"):
            cv2.putText(annotated, f"dist: {dist:.0f}px",
                        (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        cv2.imshow("DCP — Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[Fin] Programme arrêté.")


if __name__ == "__main__":
    main()
