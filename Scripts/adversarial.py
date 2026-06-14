import torch
import numpy as np
import cv2

def load_frame_as_tensor(video_path, frame_index=0, imgsz=640, device="cpu"):
    """Estrae un frame dal video e lo converte in tensore PyTorch normalizzato.

    Args:
        video_path: percorso del video.
        frame_index: indice del frame da estrarre.
        imgsz: dimensione (quadrata) a cui ridimensionare il frame.
        device: device PyTorch ("cpu", "cuda", "mps").

    Returns:
        tensor: tensore (1, 3, imgsz, imgsz), valori in [0, 1], requires_grad=True.
        frame_rgb: frame originale in RGB (per riferimento/visualizzazione).
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
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    return tensor, frame_rgb

def fgsm_attack(model, image_tensor, epsilon=0.03):
    """Esegue un attacco FGSM untargeted (hiding) su un singolo frame.

    Args:
        model: istanza YOLO (ultralytics).
        image_tensor: tensore (1, 3, H, W), requires_grad=True, valori in [0, 1].
        epsilon: intensita' della perturbazione (tipicamente 0.01 - 0.1).

    Returns:
        adv_tensor: tensore perturbato, valori in [0, 1].
        perturbation: la perturbazione applicata (per visualizzarla).
    """
    net = model.model
    net.eval()

    if image_tensor.grad is not None:
        image_tensor.grad.zero_()

    # Forward pass: output crudo (prima della NMS)
    raw_preds = net(image_tensor)
    if isinstance(raw_preds, (tuple, list)):
        raw_preds = raw_preds[0]

    # raw_preds shape attesa: (batch, 4 + num_classes, num_anchors)
    # le prime 4 righe sono le coordinate dei box, il resto le confidenze di classe.
    # Se questa riga da' errore, esegui `print(raw_preds.shape)` e adatta l'indice.
    class_scores = raw_preds[:, 4:, :]
    loss = class_scores.max(dim=1)[0].sum()

    net.zero_grad()
    loss.backward()

    grad = image_tensor.grad.data
    perturbation = -epsilon * grad.sign()  # direzione che riduce la confidenza
    adv_tensor = torch.clamp(image_tensor + perturbation, 0.0, 1.0)

    return adv_tensor.detach(), perturbation.detach()


def pgd_attack(model, image_tensor, epsilon=0.03, alpha=0.005, num_iter=10):
    """Esegue un attacco PGD (Projected Gradient Descent) untargeted (hiding).

    PGD e' la versione iterativa di FGSM: ad ogni iterazione applica un piccolo
    step nella direzione che riduce la confidenza, poi proietta il risultato
    nella palla L_inf di raggio `epsilon` centrata sull'immagine originale,
    in modo che la perturbazione totale non superi mai `epsilon`.

    Args:
        model: istanza YOLO (ultralytics).
        image_tensor: tensore (1, 3, H, W), valori in [0, 1].
        epsilon: raggio massimo della perturbazione (norma L_inf), es. 0.01 - 0.1.
        alpha: step size per ogni iterazione (tipicamente epsilon / 4 ~ epsilon / 10).
        num_iter: numero di iterazioni (piu' iterazioni = attacco piu' efficace ma piu' lento).

    Returns:
        adv_tensor: tensore perturbato, valori in [0, 1].
        perturbation: la perturbazione totale applicata (adv - originale).
    """
    net = model.model
    net.eval()

    original = image_tensor.detach().clone()
    adv = original.clone().detach()

    for _ in range(num_iter):
        adv.requires_grad_(True)

        # Forward pass: output crudo (prima della NMS)
        raw_preds = net(adv)
        if isinstance(raw_preds, (tuple, list)):
            raw_preds = raw_preds[0]

        class_scores = raw_preds[:, 4:, :]
        loss = class_scores.max(dim=1)[0].sum()

        net.zero_grad()
        loss.backward()

        grad = adv.grad.data
        adv = adv.detach() - alpha * grad.sign()  # step nella direzione che riduce la confidenza

        # Proietta la perturbazione totale nella palla L_inf di raggio epsilon
        perturbation = torch.clamp(adv - original, -epsilon, epsilon)
        adv = torch.clamp(original + perturbation, 0.0, 1.0)

    return adv.detach(), (adv - original).detach()


def tensor_to_image(t):
    """Converte un tensore (1, 3, H, W) in [0,1] in immagine numpy RGB uint8 (H, W, 3)."""
    img = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (img * 255).clip(0, 255).astype(np.uint8)