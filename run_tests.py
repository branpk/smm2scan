import json
from pathlib import Path
import sys

import numpy as np

import smm2_analyze

smm2_analyze._analyze.load_ocr()
print()

tests = sorted(Path("tests").glob("*.npy"))

args = sys.argv[1:]
if args:
    is_match = lambda file: any(arg in str(file) for arg in args)
    tests = list(filter(is_match, tests))

for i, npy_file in enumerate(tests, 1):
    name = npy_file.stem
    label = f"{name} ({i}/{len(tests)})"
    img = np.load(npy_file)

    json_file = npy_file.with_suffix(".json")
    if json_file.exists():
        with open(json_file) as f:
            test_data = json.load(f)
    else:
        test_data = {"frame_data": {}}

    exception = None
    expected = test_data["frame_data"]
    try:
        actual = smm2_analyze.analyze_frame(img)
    except Exception as e:
        exception = e
        actual = {}

    keys = list(actual)
    for key in expected:
        if key not in keys:
            keys.append(key)

    mismatches = []
    for key in keys:
        expected_value = expected.get(key, "<undefined>")
        actual_value = actual.get(key, "<undefined>")

        if expected_value != actual_value:
            mismatches.append(
                f"\x1b[1m{key}:\x1b[0m {repr(expected_value)} -> {repr(actual_value)}"
            )

    if not mismatches and not exception:
        print(f"\x1b[32m[PASS] {label}\x1b[0m")
    else:
        print(f"\x1b[31m[FAIL] {label}\x1b[0m")
        if exception:
            print(f"  \x1b[1mException:\x1b[0m {exception}")
        for mismatch in mismatches:
            print(f"  {mismatch}")

print()
