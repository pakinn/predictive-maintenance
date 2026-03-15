import re
from datetime import datetime
from pathlib import Path
import pandas as pd

def _norm_num(tok: str) -> str:
    if re.fullmatch(r'\d*\.?\d+\-\d+', tok):
        a, b = tok.split('-')
        return f"{a}e-{b}"
    if re.fullmatch(r'-\d*\.?\d+\-\d+', tok):
        a, b = tok[1:].split('-')
        return f"-{a}e-{b}"
    return tok

def parse_waveform_txt(path: str | Path) -> tuple[dict, pd.DataFrame]:
    path = Path(path)
    equipment = meas_point = dt = amp_unit = None
    data = []

    with path.open('r', errors='ignore') as f:
        for line in f:
            if 'Equipment:' in line:
                equipment = line.split('Equipment:')[-1].strip()
            if 'Meas. Point:' in line:
                meas_point = line.split('Meas. Point:')[-1].strip()
            if 'Date/Time:' in line and 'Amplitude:' in line:
                m = re.search(r'Date/Time:\s*(.*?)\s+Amplitude:\s*(.*)', line)
                if m:
                    dt, amp_unit = m.group(1).strip(), m.group(2).strip()

            if 'Time' in line or '----' in line or '*' in line:
                continue

            tokens = [_norm_num(t) for t in line.strip().split()]
            nums = []
            for tok in tokens:
                if re.fullmatch(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?', tok, re.I):
                    nums.append(float(tok))
            if len(nums) >= 2 and len(nums) % 2 == 0:
                data.extend(list(zip(nums[::2], nums[1::2])))

    df = (pd.DataFrame(data, columns=['time_ms', 'amplitude_g'])
            .sort_values('time_ms')
            .reset_index(drop=True))

    metadata = {
        'file': path.name,
        'equipment_header': equipment,
        'meas_point': meas_point,
        'header_datetime': (
            datetime.strptime(dt, '%d-%b-%y %H:%M:%S').isoformat() if dt else None
        ),
        'amplitude_unit': amp_unit,
        'n_samples': len(df),
    }
    return metadata, df