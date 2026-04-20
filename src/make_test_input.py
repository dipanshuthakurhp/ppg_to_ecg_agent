from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

def csv_to_test_input_c(csv_path: str, out_c_path: str, col: str = "A1", n: int = 250) -> str:
    df = pd.read_csv(csv_path)

    want = col.strip().lower()
    colname = None
    for name in df.columns:
        if str(name).strip().lower() == want:
            colname = name
            break
    if colname is None:
        raise ValueError(f"Column '{col}' not found. Columns: {df.columns.tolist()}")

    sig = df[colname].dropna().to_numpy()
    if sig.size < n:
        raise ValueError(f"Column '{col}' has {sig.size} samples; need at least {n}")

    segment = sig[:n].astype(np.float32)
    min_val = np.min(segment)
    max_val = np.max(segment)

    segment = (((segment-min_val)/(max_val-min_val))*2)-1
    scaled = (segment) * 127
    scaled = np.clip(scaled, -128, 127).astype(np.int8)

    out = Path(out_c_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        f.write("#include <stdint.h>\n\n")
        f.write(f"const int8_t test_input[{n}] = {{\n  ")
        for i, v in enumerate(scaled.tolist()):
            f.write(f"{int(v)}, ")
            if (i + 1) % 16 == 0:
                f.write("\n  ")
        f.write("\n};\n")
    return str(out)

# csv_to_test_input_c("D:\physiofusion_agent_v2\data\ppg\ppg_1.csv", "D:\esp32_firmware\main\\test_input.c", "A1")