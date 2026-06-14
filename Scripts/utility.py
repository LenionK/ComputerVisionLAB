def print_classes_table(model, n_cols=4, sort_alpha=True):
    """Stampa le classi del modello in colonne multiple.

    Args:
        model: istanza YOLO (usa model.names).
        n_cols: numero di colonne da usare.
        sort_alpha: se True ordina le classi alfabeticamente, 
                    altrimenti per ID numerico.
    """
    items = list(model.names.items())
    if sort_alpha:
        items = sorted(items, key=lambda x: x[1])

    n_rows = -(-len(items) // n_cols)  # divisione arrotondata verso l'alto

    for row in range(n_rows):
        line = ""
        for col in range(n_cols):
            idx_item = row + col * n_rows
            if idx_item < len(items):
                idx, name = items[idx_item]
                line += f"{idx:>3} | {name:<18}"
        print(line)