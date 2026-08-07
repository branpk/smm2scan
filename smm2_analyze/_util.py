import cv2
import numpy as np

template_cache: dict[str, np.ndarray] = {}


def matches_template(
    img: np.ndarray,
    template_name: str,
    pixel_threshold: float = 40.0,
    percent_threshold: float = 0.8,
) -> bool:
    template = template_cache.get(template_name)
    if template is None:
        template = cv2.cvtColor(
            cv2.imread(f"templates/{template_name}.png"), cv2.COLOR_BGR2RGB  # type: ignore
        ).astype(np.float32)
        template_cache[template_name] = template

    mask = (template != [169, 69, 169]).all(axis=-1)
    matches = np.abs(img - template.astype(np.float32)).max(axis=-1) < pixel_threshold
    percent = matches[mask].mean()
    return bool(percent > percent_threshold)


__all__ = []
