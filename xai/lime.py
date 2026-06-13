# xai/lime.py

import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries


def explain_with_lime(model, image, transform, num_samples=1000):
    """
    Generate LIME explanation for a single image.
    """

    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        images = torch.stack([transform(img) for img in images])
        with torch.no_grad():
            outputs = model(images)
        return outputs.softmax(dim=1).cpu().numpy()

    explanation = explainer.explain_instance(
        np.array(image),
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
    )

    return explanation
