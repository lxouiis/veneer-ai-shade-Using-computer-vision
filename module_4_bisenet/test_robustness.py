# Robustness spot-check for BiSeNet face parsing: instead of random
# CelebAMask-HQ samples (which are mostly clean, well-lit, frontal photos),
# this deliberately picks harder cases using the dataset's own attribute
# labels - a reasonable stand-in for the non-ideal conditions real patient
# photos will have (motion blur, partial occlusion from glasses/hats).
#
# Assumes model.py has already been %%writefile'd to disk and parse_face.py
# has already been run once in this session (same as the normal module_4
# flow) - it reuses those same functions/constants.

import os
import random
import matplotlib.pyplot as plt

CELEBA_ROOT = "/kaggle/input/datasets/ipythonx/celebamaskhq/CelebAMask-HQ"
ATTR_FILE = os.path.join(CELEBA_ROOT, "CelebAMask-HQ-attribute-anno.txt")

# Which attributes to test, and how many images to sample per attribute.
HARD_CASE_ATTRIBUTES = ["Blurry", "Eyeglasses", "Wearing_Hat"]
NUM_PER_ATTRIBUTE = 2
RANDOM_SEED = 42


def parse_attributes(attr_file):
    with open(attr_file, "r") as f:
        lines = f.read().splitlines()

    # Standard CelebA attribute file format: line 0 = image count,
    # line 1 = space-separated attribute names, remaining lines =
    # "<filename> <val1> <val2> ..." with values encoded as 1 / -1.
    attr_names = lines[1].split()
    records = {}
    for line in lines[2:]:
        parts = line.split()
        filename = parts[0]
        values = [int(v) for v in parts[1:]]
        records[filename] = dict(zip(attr_names, values))
    return records


def select_hard_cases(records, attribute, n, seed=RANDOM_SEED):
    matches = [fname for fname, attrs in records.items() if attrs.get(attribute) == 1]
    rng = random.Random(seed)
    return rng.sample(matches, min(n, len(matches)))


if __name__ == "__main__":
    print(f"Parsing attribute file: {ATTR_FILE}")
    records = parse_attributes(ATTR_FILE)
    print(f"Loaded attributes for {len(records)} images")

    model = load_bisenet_model(BISENET_WEIGHTS)  # noqa: F821 - defined by parse_face.py, already run this session

    results = []
    for attribute in HARD_CASE_ATTRIBUTES:
        test_images = select_hard_cases(records, attribute, NUM_PER_ATTRIBUTE)
        print(f"\n{'=' * 70}\nTesting attribute: {attribute} ({len(test_images)} images)\n{'=' * 70}")

        for filename in test_images:
            image_path = os.path.join(CELEBA_ROOT, "CelebA-HQ-img", filename)
            if not os.path.exists(image_path):
                print(f"  {filename}: image file not found, skipping")
                continue

            image_tensor, original_image = preprocess_for_bisenet(image_path)  # noqa: F821
            parsing_map = run_face_parsing(model, image_tensor)  # noqa: F821

            skin_crop, skin_mask = extract_region_crop(original_image, parsing_map, SKIN_LABELS, "skin")  # noqa: F821
            lip_crop, lip_mask = extract_region_crop(original_image, parsing_map, LIP_LABELS, "lip")  # noqa: F821

            skin_px = int(skin_mask.sum()) if skin_mask is not None else 0
            lip_px = int(lip_mask.sum()) if lip_mask is not None else 0
            print(f"  {filename} ({attribute}): skin_found={skin_crop is not None} "
                  f"({skin_px} px), lip_found={lip_crop is not None} ({lip_px} px)")

            results.append((filename, attribute, skin_crop, lip_crop, original_image, parsing_map))

    # Visualize everything in one grid: 4 columns (original, parsing map,
    # skin crop, lip crop) x however many hard-case images we tested.
    fig, axes = plt.subplots(len(results), 4, figsize=(16, 4 * len(results)))
    if len(results) == 1:
        axes = [axes]

    for row, (filename, attribute, skin_crop, lip_crop, original_image, parsing_map) in zip(axes, results):
        row[0].imshow(original_image)
        row[0].set_title(f"{filename} ({attribute})")
        row[0].axis("off")

        row[1].imshow(parsing_map, cmap="tab20")
        row[1].set_title("Parsing Map")
        row[1].axis("off")

        row[2].imshow(skin_crop if skin_crop is not None else parsing_map * 0)
        row[2].set_title("Skin Crop" if skin_crop is not None else "Skin (not found)")
        row[2].axis("off")

        row[3].imshow(lip_crop if lip_crop is not None else parsing_map * 0)
        row[3].set_title("Lip Crop" if lip_crop is not None else "Lip (not found)")
        row[3].axis("off")

    plt.tight_layout()
    plt.savefig("/kaggle/working/outputs/robustness_check.png")
    plt.show()
