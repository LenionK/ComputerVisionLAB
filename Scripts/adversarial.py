"""
adversarial.py — Attacchi adversarial white-box per modelli YOLO (Ultralytics).

Supporta:
  - YOLOv8  (es. yolov8n.pt)
  - YOLOv11 (es. yolo11n.pt)

Entrambi condividono la stessa struttura di output della detection head:
  raw_preds shape: (batch, 4 + num_classes, num_anchors)
Il codice degli attacchi FGSM e PGD è quindi identico per i due modelli;
basta passare l'istanza YOLO corretta al parametro `model`.

Attacchi implementati:
  - FGSM  : Fast Gradient Sign Method (single-step)
  - PGD   : Projected Gradient Descent (multi-step, L_inf)

Tipo di attacco: white-box, untargeted, hiding
  (obiettivo: far scomparire le detection riducendo le confidenze di classe).
"""

import torch
import numpy as np
import cv2
import os


# ─── Utility interna ──────────────────────────────────────────────────────────

def _detach_inference_buffers(module):
    """Clona tensori in 'inference mode' cachati come attributi nei sotto-moduli.

    Ultralytics YOLO calcola e mette in cache alcuni tensori (es. `anchors`,
    `strides` nella detection head) durante le normali chiamate `model(img)`,
    che avvengono in `torch.inference_mode()`. Questi tensori non possono essere
    usati nel grafo di autograd (errore: "Inference tensors cannot be saved for
    backward").

    Soluzione: attraversa ricorsivamente tutti i sotto-moduli e clona ogni
    tensore in inference mode, rendendolo un tensore normale utilizzabile con
    autograd. Compatibile con YOLOv8 e YOLOv11.
    """
    for m in module.modules():
        for name, val in vars(m).items():
            if isinstance(val, torch.Tensor) and val.is_inference():
                setattr(m, name, val.clone())


def _get_raw_preds(net, image_tensor):
    """Esegue il forward pass e restituisce il tensore di predizioni grezze.

    Args:
        net   : modello PyTorch interno (model.model per istanze Ultralytics).
        image_tensor: tensore (1, 3, H, W), valori in [0, 1], requires_grad=True.

    Returns:
        raw_preds: tensore (batch, 4 + num_classes, num_anchors).

    Note:
        - YOLOv8 e YOLOv11 restituiscono entrambi una tupla/lista di cui il
          primo elemento è il tensore delle predizioni grezze (pre-NMS).
        - Se il modello restituisce una struttura diversa, stampare
          `type(raw_preds)` e `raw_preds[0].shape` per adattare.
    """
    raw_preds = net(image_tensor)
    if isinstance(raw_preds, (tuple, list)):
        raw_preds = raw_preds[0]
    return raw_preds


# ─── Caricamento frame ────────────────────────────────────────────────────────

def load_frame_as_tensor(video_path, frame_index=0, imgsz=640, device="cpu"):
    """Estrae un frame dal video e lo converte in tensore PyTorch normalizzato.

    Args:
        video_path  : percorso del video.
        frame_index : indice del frame da estrarre (0-based).
        imgsz       : dimensione (quadrata) a cui ridimensionare il frame.
        device      : device PyTorch ("cpu", "cuda", "mps").

    Returns:
        tensor    : tensore (1, 3, imgsz, imgsz), valori in [0, 1],
                    requires_grad=True.
        frame_rgb : frame originale in RGB (per riferimento/visualizzazione).
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError(f"Impossibile leggere il frame {frame_index} da {video_path}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (imgsz, imgsz))

    img = resized.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)   # HWC → CHW
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    return tensor, frame_rgb


# ─── Attacchi adversarial ─────────────────────────────────────────────────────

def fgsm_attack(model, image_tensor, epsilon=0.03):
    """Attacco FGSM untargeted (hiding) su un singolo frame.

    Esegue un singolo step di perturbazione nella direzione che minimizza le
    confidenze di classe (hiding). Compatibile con YOLOv8n e YOLOv11n.

    Args:
        model        : istanza YOLO Ultralytics (YOLOv8 o YOLOv11).
        image_tensor : tensore (1, 3, H, W), requires_grad=True, valori in [0, 1].
        epsilon      : intensità della perturbazione (tipicamente 0.01 – 0.1).

    Returns:
        adv_tensor   : tensore perturbato, valori in [0, 1].
        perturbation : perturbazione applicata (utile per visualizzazione).
    """
    net = model.model
    net.eval()
    _detach_inference_buffers(net)

    if image_tensor.grad is not None:
        image_tensor.grad.zero_()

    raw_preds = _get_raw_preds(net, image_tensor)

    # Le prime 4 righe sono le coordinate dei box; le restanti sono i class scores.
    # Questo vale per YOLOv8 e YOLOv11 (stessa struttura di head).
    class_scores = raw_preds[:, 4:, :]
    loss = class_scores.max(dim=1)[0].sum()

    net.zero_grad()
    loss.backward()

    grad = image_tensor.grad.data
    perturbation = -epsilon * grad.sign()   # segno negativo → riduce la confidenza
    adv_tensor = torch.clamp(image_tensor + perturbation, 0.0, 1.0)

    return adv_tensor.detach(), perturbation.detach()


def pgd_attack(model, image_tensor, epsilon=0.03, alpha=0.005, num_iter=10):
    """Attacco PGD untargeted (hiding) su un singolo frame.

    Versione iterativa di FGSM: ad ogni iterazione applica un piccolo step
    nella direzione che riduce le confidenze, poi proietta il risultato nella
    palla L_inf di raggio `epsilon` centrata sull'immagine originale.
    Compatibile con YOLOv8n e YOLOv11n.

    Args:
        model        : istanza YOLO Ultralytics (YOLOv8 o YOLOv11).
        image_tensor : tensore (1, 3, H, W), valori in [0, 1].
        epsilon      : raggio massimo della perturbazione (norma L_inf).
        alpha        : step size per ogni iterazione (tipicamente epsilon/4 – epsilon/10).
        num_iter     : numero di iterazioni.

    Returns:
        adv_tensor   : tensore perturbato, valori in [0, 1].
        perturbation : perturbazione totale applicata (adv - originale).
    """
    net = model.model
    net.eval()
    _detach_inference_buffers(net)

    original = image_tensor.detach().clone()
    adv = original.clone().detach()

    for _ in range(num_iter):
        adv.requires_grad_(True)

        raw_preds = _get_raw_preds(net, adv)

        class_scores = raw_preds[:, 4:, :]
        loss = class_scores.max(dim=1)[0].sum()

        net.zero_grad()
        loss.backward()

        grad = adv.grad.data
        adv = adv.detach() - alpha * grad.sign()

        # Proiezione nella palla L_inf di raggio epsilon
        perturbation = torch.clamp(adv - original, -epsilon, epsilon)
        adv = torch.clamp(original + perturbation, 0.0, 1.0)

    return adv.detach(), (adv - original).detach()


# ─── Utility di conversione ───────────────────────────────────────────────────

def tensor_to_image(t):
    """Converte un tensore (1, 3, H, W) in [0,1] in immagine numpy RGB uint8."""
    img = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (img * 255).clip(0, 255).astype(np.uint8)


# ─── Pipeline video adversarial ───────────────────────────────────────────────

def run_adversarial_video_pgd(
    source,
    output,
    yolo_model,
    epsilon=0.03,
    alpha=0.005,
    num_iter=10,
    conf=0.25,
    device="cpu",
):
    """Applica PGD frame-by-frame a un video e salva il risultato annotato.

    Funziona con qualsiasi modello Ultralytics YOLO (YOLOv8, YOLOv11, ecc.).
    Passare il tensore PyTorch float direttamente a YOLO preserva la
    perturbazione adversarial senza le distorsioni della conversione uint8.

    Args:
        source     : percorso del video di input.
        output     : percorso del video di output.
        yolo_model : istanza YOLO Ultralytics già caricata
                     (es. YOLO('yolov8n.pt') oppure YOLO('yolo11n.pt')).
        epsilon    : intensità massima della perturbazione PGD.
        alpha      : step size PGD per iterazione.
        num_iter   : numero di iterazioni PGD.
        conf       : soglia di confidenza per la detection.
        device     : device PyTorch ("cpu", "cuda", "mps").
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Impossibile aprire il video sorgente: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    imgsz = 640  # dimensione nativa d'attacco

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output, fourcc, fps, (imgsz, imgsz))

    model_name = getattr(yolo_model, "ckpt_path", "modello YOLO")
    print(f"Inizio elaborazione video con PGD — modello: {model_name}")
    print(f"  epsilon={epsilon}, alpha={alpha}, num_iter={num_iter}, conf={conf}")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"  Frame {frame_count}...")

        # 1. Prepara il frame a 640×640 in RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (imgsz, imgsz))

        # 2. Tensore float [0, 1]
        img_np = resized.astype(np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)   # HWC → CHW
        image_tensor = torch.from_numpy(img_np).unsqueeze(0).float().to(device)

        # 3. Attacco PGD
        adv_tensor, _ = pgd_attack(
            yolo_model, image_tensor,
            epsilon=epsilon, alpha=alpha, num_iter=num_iter,
        )

        # 4. Inferenza DIRETTAMENTE sul tensore perturbato (no conversione uint8!)
        #    YOLOv8 e YOLOv11 accettano tensori float [0,1] di shape (1, 3, 640, 640).
        results = yolo_model(adv_tensor, conf=conf, verbose=False)

        # 5. Conversione solo per visualizzazione
        adv_img = adv_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        adv_img = (adv_img * 255.0).astype(np.uint8)
        frame_vis = cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR)

        # 6. Disegna le (eventuali) box sul frame perturbato
        results[0].orig_img = frame_vis
        annotated_frame = results[0].plot()

        writer.write(annotated_frame)

    cap.release()
    writer.release()
    print(f"Video salvato in: {output}")
    print(f"Totale frame elaborati: {frame_count}")


def run_adversarial_video_fgsm(
    source,
    output,
    yolo_model,
    epsilon=0.03,
    conf=0.25,
    device="cpu",
):
    """Applica FGSM frame-by-frame a un video e salva il risultato annotato.

    Versione single-step (più veloce di PGD, generalmente meno efficace).
    Compatibile con YOLOv8n e YOLOv11n.

    Args:
        source     : percorso del video di input.
        output     : percorso del video di output.
        yolo_model : istanza YOLO Ultralytics già caricata.
        epsilon    : intensità della perturbazione FGSM.
        conf       : soglia di confidenza per la detection.
        device     : device PyTorch ("cpu", "cuda", "mps").
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Impossibile aprire il video sorgente: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    imgsz = 640

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output, fourcc, fps, (imgsz, imgsz))

    model_name = getattr(yolo_model, "ckpt_path", "modello YOLO")
    print(f"Inizio elaborazione video con FGSM — modello: {model_name}")
    print(f"  epsilon={epsilon}, conf={conf}")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"  Frame {frame_count}...")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (imgsz, imgsz))

        img_np = resized.astype(np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)
        image_tensor = torch.from_numpy(img_np).unsqueeze(0).float().to(device)
        image_tensor.requires_grad_(True)

        adv_tensor, _ = fgsm_attack(yolo_model, image_tensor, epsilon=epsilon)

        results = yolo_model(adv_tensor, conf=conf, verbose=False)

        adv_img = adv_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        adv_img = (adv_img * 255.0).astype(np.uint8)
        frame_vis = cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR)

        results[0].orig_img = frame_vis
        annotated_frame = results[0].plot()

        writer.write(annotated_frame)

    cap.release()
    writer.release()
    print(f"Video salvato in: {output}")
    print(f"Totale frame elaborati: {frame_count}")