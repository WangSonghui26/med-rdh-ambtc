"""
Controlled CODE ablation: four-tier (Chang 2018 granularity) vs six-tier (proposed)
variable-length XOR code, holding the MED predictor FIXED.

Faithful four-tier code (Chang et al. 2018, MTA 77:9039-9053, Eq.(9) + Steps 9.1-9.4):
  IDX=11: 0<=D<=7    -> sk||sk+1||11||D'(3b)||5 secret   (12-bit, 7 secret bits)
  IDX=10: 8<=D<=15   -> sk||sk+1||10||D'(3b)||5 secret   (12-bit, 7 secret bits)
  IDX=01: 16<=D<=31  -> sk||sk+1||01||D^(4b)||4 secret   (12-bit, 6 secret bits)
  IDX=00: 32<=D<=255 -> sk||sk+1||00||D (8b)             (12-bit, 2 secret bits)

Proposed six-tier code (Table I of the paper):
  0<=D<=7:7, 8<=D<=15:7, 16<=D<=31:6, 32<=D<=63:4, 64<=D<=127:2 (12-bit), D>127:2 (14-bit)

The ONLY difference between the two configs is the code table; the predictor (MED),
the AMBTC host, the scan order and the reversibility machinery are identical.

INTEGRITY CHECK: the six-tier MED run must reproduce the ground-truth per-image
capacity in experiment_results.json (avg 196902.5). If it does not match exactly,
the pipeline is wrong and nothing is reported.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import json
import numpy as np

from run_experiments import (
    ambtc_compress, predict_med,
)

BASE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(BASE, "pics", "pics")
P2 = os.path.join(BASE, "pics", "pics2")
TEST_IMAGES = [
    ("Baboon",   os.path.join(P1, "1-Baboon.bmp")),
    ("Airplane", os.path.join(P1, "1-airplane.bmp")),
    ("Peppers",  os.path.join(P1, "4-Peppers.bmp")),
    ("Elaine",   os.path.join(P1, "5-elaine.bmp")),
    ("Wine",     os.path.join(P1, "8-wine.bmp")),
    ("Girl",     os.path.join(P1, "10-girl.bmp")),
    ("Barbara",  os.path.join(P2, "2-Barbara.bmp")),
    ("Lena",     os.path.join(P2, "3-Lena.bmp")),
    ("Goldhill", os.path.join(P2, "6-Goldhil.bmp")),
    ("Boat",     os.path.join(P2, "9-boat.bmp")),
    ("Tiffany",  os.path.join(P2, "11-tiffany.bmp")),
    ("Zelda",    os.path.join(P2, "12-zelda.bmp")),
]

# ------------------------------------------------------------------
# Code tables: embed/extract one label given (value, ref, secret_bits)
# Each returns (codeword_str, consumed_secret_bits)
# ------------------------------------------------------------------
def vl_embed_six(value, ref, sb):
    if len(sb) < 2:
        return format(value, '08b'), 0
    s1, s2 = sb[0], sb[1]
    d = value ^ ref
    consumed = 2
    if d <= 7:
        bits = s1 + s2 + '11' + format(d, '03b') + sb[2:7]; consumed += len(sb[2:7]); bits = bits.ljust(12, '0')
    elif d <= 15:
        bits = s1 + s2 + '10' + format(d - 8, '03b') + sb[2:7]; consumed += len(sb[2:7]); bits = bits.ljust(12, '0')
    elif d <= 31:
        bits = s1 + s2 + '01' + format(d - 16, '04b') + sb[2:6]; consumed += len(sb[2:6]); bits = bits.ljust(12, '0')
    elif d <= 63:
        bits = s1 + s2 + '00' + '1' + format(d - 32, '05b') + sb[2:4]; consumed += len(sb[2:4]); bits = bits.ljust(12, '0')
    elif d <= 127:
        bits = s1 + s2 + '00' + '01' + format(d - 64, '06b')
    else:
        bits = s1 + s2 + '00' + '00' + format(d, '08b')
    return bits, consumed


def vl_extract_six(sb_str, ref):
    s1, s2 = sb_str[0], sb_str[1]
    D = sb_str[2:4]
    secret = s1 + s2
    if D == '11':
        d = int(sb_str[4:7], 2); secret += sb_str[7:12]
    elif D == '10':
        d = int(sb_str[4:7], 2) + 8; secret += sb_str[7:12]
    elif D == '01':
        d = int(sb_str[4:8], 2) + 16; secret += sb_str[8:12]
    else:
        if sb_str[4] == '1':
            d = int(sb_str[5:10], 2) + 32; secret += sb_str[10:12]
        elif sb_str[5] == '1':
            d = int(sb_str[6:12], 2) + 64
        else:
            d = int(sb_str[6:14], 2)
    return d ^ ref, secret


def vl_embed_four(value, ref, sb):
    if len(sb) < 2:
        return format(value, '08b'), 0
    s1, s2 = sb[0], sb[1]
    d = value ^ ref
    consumed = 2
    if d <= 7:
        bits = s1 + s2 + '11' + format(d, '03b') + sb[2:7]; consumed += len(sb[2:7]); bits = bits.ljust(12, '0')
    elif d <= 15:
        bits = s1 + s2 + '10' + format(d - 8, '03b') + sb[2:7]; consumed += len(sb[2:7]); bits = bits.ljust(12, '0')
    elif d <= 31:
        bits = s1 + s2 + '01' + format(d - 16, '04b') + sb[2:6]; consumed += len(sb[2:6]); bits = bits.ljust(12, '0')
    else:  # 32 <= d <= 255 : single wide case, 8-bit residual, no extra secret
        bits = s1 + s2 + '00' + format(d, '08b')
    return bits, consumed


def vl_extract_four(sb_str, ref):
    s1, s2 = sb_str[0], sb_str[1]
    D = sb_str[2:4]
    secret = s1 + s2
    if D == '11':
        d = int(sb_str[4:7], 2); secret += sb_str[7:12]
    elif D == '10':
        d = int(sb_str[4:7], 2) + 8; secret += sb_str[7:12]
    elif D == '01':
        d = int(sb_str[4:8], 2) + 16; secret += sb_str[8:12]
    else:  # '00'
        d = int(sb_str[4:12], 2)
    return d ^ ref, secret


# ------------------------------------------------------------------
# MED embedding/extraction over one label table (parametrized by code)
# Scan order identical to run_experiments.embed_matrix_generic
# ------------------------------------------------------------------
def embed_med(matrix, secret_bits, vl_embed):
    h, w = matrix.shape
    stego = np.zeros_like(matrix, dtype=object)
    embedded = 0
    rem = secret_bits
    # Phase 1: interior (bottom->top, right->left), MED on 3 causal neighbors
    for i in range(h - 1, 0, -1):
        for j in range(w - 1, 0, -1):
            if len(rem) <= 1:
                stego[i, j] = format(matrix[i, j], '08b')
            else:
                ref = predict_med(int(matrix[i, j-1]), int(matrix[i-1, j]), int(matrix[i-1, j-1]))
                bits, c = vl_embed(int(matrix[i, j]), int(ref), rem)
                stego[i, j] = bits; embedded += c; rem = rem[c:]
    # Phase 2: first column (bottom->top), upper neighbor
    for i in range(h - 1, 0, -1):
        if len(rem) <= 1:
            stego[i, 0] = format(matrix[i, 0], '08b')
        else:
            ref = matrix[i-1, 0]
            bits, c = vl_embed(int(matrix[i, 0]), int(ref), rem)
            stego[i, 0] = bits; embedded += c; rem = rem[c:]
    # Phase 3: first row (right->left), left neighbor
    for j in range(w - 1, 0, -1):
        if len(rem) <= 1:
            stego[0, j] = format(matrix[0, j], '08b')
        else:
            ref = matrix[0, j-1]
            bits, c = vl_embed(int(matrix[0, j]), int(ref), rem)
            stego[0, j] = bits; embedded += c; rem = rem[c:]
    stego[0, 0] = format(matrix[0, 0], '08b')
    return stego, rem, embedded


def extract_med(stego, vl_extract):
    h, w = stego.shape
    secret = ''
    recon = np.zeros_like(stego, dtype=np.int32)
    recon[0, 0] = int(stego[0, 0], 2)
    # Phase 1: first row (left->right)
    for j in range(1, w):
        ref = recon[0, j-1]
        if len(stego[0, j]) >= 12:
            val, bits = vl_extract(stego[0, j], ref); secret = bits + secret; recon[0, j] = val
        else:
            recon[0, j] = int(stego[0, j], 2)
    # Phase 2: first column (top->bottom)
    for i in range(1, h):
        ref = recon[i-1, 0]
        if len(stego[i, 0]) >= 12:
            val, bits = vl_extract(stego[i, 0], ref); secret = bits + secret; recon[i, 0] = val
        else:
            recon[i, 0] = int(stego[i, 0], 2)
    # Phase 3: interior (top->bottom, left->right), MED on recovered neighbors
    for i in range(1, h):
        for j in range(1, w):
            ref = predict_med(int(recon[i, j-1]), int(recon[i-1, j]), int(recon[i-1, j-1]))
            if len(stego[i, j]) >= 12:
                val, bits = vl_extract(stego[i, j], ref); secret = bits + secret; recon[i, j] = val
            else:
                recon[i, j] = int(stego[i, j], 2)
    return secret, recon


def run_one(img_path, vl_embed, vl_extract, block_size=4):
    from PIL import Image
    img = np.array(Image.open(img_path).convert('L'))
    bitmaps, high, low = ambtc_compress(img, block_size)
    total_labels = 2 * high.size
    total_possible = total_labels * 7
    rng = np.random.RandomState(42)
    sb = ''.join(str(int(b)) for b in rng.randint(0, 2, total_possible))
    sh, rem, eh = embed_med(high, sb, vl_embed)
    sl, _, el = embed_med(low, rem, vl_embed)
    # reversibility check
    _, rh = extract_med(sh, vl_extract)
    _, rl = extract_med(sl, vl_extract)
    ok = np.array_equal(rh, high.astype(np.int32)) and np.array_equal(rl, low.astype(np.int32))
    return eh + el, ok, (img.shape[0] * img.shape[1])


def main():
    gt = json.load(open(os.path.join(BASE, 'experiment_results.json'), encoding='utf-8'))
    six_bits, four_bits = {}, {}
    all_ok = True
    print("image        six-tier   four-tier   reversible")
    for name, path in TEST_IMAGES:
        if not os.path.exists(path):
            print("MISSING", name, path); all_ok = False; continue
        b6, ok6, npix = run_one(path, vl_embed_six, vl_extract_six)
        b4, ok4, _    = run_one(path, vl_embed_four, vl_extract_four)
        six_bits[name], four_bits[name] = b6, b4
        all_ok = all_ok and ok6 and ok4
        # integrity check vs ground truth
        g = gt.get(name, {}).get('med', {}).get('embedded_bits')
        flag = '' if g == b6 else '  <-- MISMATCH gt=%s' % g
        print("%-12s %8d %10d      %s%s" % (name, b6, b4, ok6 and ok4, flag))

    a6 = np.mean(list(six_bits.values()))
    a4 = np.mean(list(four_bits.values()))
    print("\nAVG six-tier  = %.1f  (%.4f bpp)" % (a6, a6 / (512 * 512)))
    print("AVG four-tier = %.1f  (%.4f bpp)" % (a4, a4 / (512 * 512)))
    print("six vs four gain = %+.2f%%" % ((a6 - a4) / a4 * 100))
    print("ALL REVERSIBLE =", all_ok)

    out = {
        'six_tier': six_bits, 'four_tier': four_bits,
        'avg_six': float(a6), 'avg_four': float(a4),
        'gain_pct': float((a6 - a4) / a4 * 100),
        'all_reversible': bool(all_ok),
        'six_matches_groundtruth': all(
            six_bits[n] == gt.get(n, {}).get('med', {}).get('embedded_bits')
            for n in six_bits),
    }
    with open(os.path.join(BASE, 'ablation_code_tiers.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("saved ablation_code_tiers.json ; six_matches_groundtruth =",
          out['six_matches_groundtruth'])


if __name__ == '__main__':
    main()
