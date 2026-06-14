import cv2

def run_inference(source, output, model, conf=0.25, max_preview_frames=0):
    """Esegue object detection sul video e salva il risultato.

    Args:
        source: percorso del video di input.
        output: percorso del video di output.
        model: istanza YOLO già caricata.
        conf: soglia di confidenza minima.
        max_preview_frames: se > 0, restituisce i primi N frame annotati
            (come array numpy RGB) per visualizzarli nel notebook.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Impossibile aprire il video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    preview_frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)
        annotated_frame = results[0].plot()

        writer.write(annotated_frame)

        if max_preview_frames and frame_count < max_preview_frames:
            preview_frames.append(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frame elaborati: {frame_count}")

    cap.release()
    writer.release()

    print(f"Completato! Video salvato")
    return preview_frames