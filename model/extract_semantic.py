import torch
from modules.prompt_utils import generate_entity_tokens

def extract_semantic_vectors(dataset_name, class_names, text_encoder, evg_model, device):
    """
    Generate semantic vectors for each class using EVG + TextEncoder.

    Args:
        dataset_name (str): e.g., 'MiniImageNet', 'CIFAR-FS'
        class_names (List[str])
        text_encoder (TextEncoder): frozen model
        evg_model (EVGNetwork or MultiHeadEVGNetwork): trainable

    Returns:
        entity_vectors (Tensor): shape [N, D] (output of EVG)
    """
    entity_vectors = []

    evg_model.train()
    text_encoder.eval()

    for cls_name in class_names:
        entity_tokens = generate_entity_tokens(dataset_name, cls_name)

        with torch.no_grad():
            token_embeddings = text_encoder(entity_tokens).to(device)
            class_embedding = text_encoder([cls_name])[0].to(device)

        entity_vector = evg_model(class_embedding, token_embeddings)
        entity_vectors.append(entity_vector)

    entity_vectors = torch.stack(entity_vectors, dim=0)
    
    return entity_vectors
