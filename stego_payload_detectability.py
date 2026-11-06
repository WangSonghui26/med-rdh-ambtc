"""
Compressed-domain payload-detectability experiment (REAL data, no fabrication).

Threat model: a passive attacker who intercepts the transmitted AMBTC codeword
stream and runs a statistical steganalyzer on the SECRET-BIT FIELD carried inside
the 12/14-bit codewords (the residual field depends only on the labels, not on the
payload, so the payload field is the only place a leak can appear).

Detector (two standard passive tests):
  (1) chi-square bit-balance test on the payload field (H0: ones == 50%), dof=1,
      critical value 3.841 at p=0.05.
  (2) order-1 Markov entropy H1 of the payload bitstream (max 2.000 bits for an
      ideal uniform iid source); a biased/structured payload has H1 < 2.

We embed the SAME real structured message (ASCII text -> bits, top bit always 0,
hence strongly biased) in two ways across all 12 images with the MED + six-tier
code:
  (a) RAW      : text bits embedded directly (no whitening).
  (b) WHITENED : text bits XOR-ed with a keystream from a fixed PRNG seed
                 (the standard "encrypt before embed" practice).

Expected honest finding: RAW is trivially detected (chi2 >> 3.841, H1 << 2),
WHITENED is not (chi2 near threshold, ones ~= 0.5, H1 ~= 2). This shows the
scheme provides NO inherent payload confidentiality by itself -- confidentiality
comes from whitening -- and quantifies exactly how much a passive detector gains.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, json
import numpy as np
from PIL import Image

from run_experiments import ambtc_compress, predict_med
from ablation_code_tiers import embed_med, vl_embed_six, TEST_IMAGES

BASE = os.path.dirname(os.path.abspath(__file__))

# ---- real structured payload: ASCII text of the paper's own abstract ----
TEXT = ("Reversible data hiding in absolute moment block truncation coding must "
        "embed a secret payload while permitting exact recovery of the cover. "
        "We recast AMBTC RDH as predictive source coding and replace secret-driven "
        "neighbor selection with a deterministic median-edge-detection predictor. ")
def text_to_bits(s, nbits):
    b = ''.join(format(ord(c), '08b') for c in s)
    rep = (nbits // len(b)) + 1
    return (b * rep)[:nbits]

def chi2_balance(bits):
    ones = sum(1 for c in bits if c == '1')
    n = len(bits); zeros = n - ones
    e = n / 2.0
    chi2 = (ones - e) ** 2 / e + (zeros - e) ** 2 / e
    return ones / n, chi2

def order1_entropy(bits):
    # bigram counts over 4 categories
    idx = [int(c) for c in bits]
    a = np.array(idx, dtype=np.int8)
    bg = a[:-1] * 2 + a[1:]
    counts = np.bincount(bg, minlength=4).astype(float)
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log2(p)))

def embed_and_collect(vl_embed, payload_bits, block_size=4):
    """Embed payload into high then low table of every image; return the actual
    payload field that landed in the codewords (== consumed prefix), per image."""
    fields = []
    for name, path in TEST_IMAGES:
        if not os.path.exists(path):
            continue
        img = np.array(Image.open(path).convert('L'))
        _, high, low = ambtc_compress(img, block_size)
        rem = payload_bits
        _, rem2, eh = embed_med(high, rem, vl_embed)
        _, _, el = embed_med(low, rem2, vl_embed)
        consumed = eh + el
        fields.append(payload_bits[:consumed])
    return fields

def main():
    # size the payload generously so it never runs dry (>= max capacity ~230k bits)
    NB = 2 * 16384 * 7
    raw = text_to_bits(TEXT, NB)
    rng = np.random.RandomState(2024)
    keystream = ''.join(str(int(b)) for b in rng.randint(0, 2, NB))
    white = ''.join('1' if (a != b) else '0' for a, b in zip(raw, keystream))

    fo, c2 = chi2_balance(raw); h1 = order1_entropy(raw)
    print("RAW text stream   : ones=%.4f  chi2=%.1f  H1=%.4f bits" % (fo, c2, h1))
    fo, c2 = chi2_balance(white); h1w = order1_entropy(white)
    print("WHITENED stream   : ones=%.4f  chi2=%.1f  H1=%.4f bits" % (fo, c2, h1w))

    # Now measure the ACTUAL embedded payload field per condition
    summary = {}
    for label, bits in [('raw', raw), ('whitened', white)]:
        fields = embed_and_collect(vl_embed_six, bits)
        ones = []; chi2s = []; h1s = []; lens = []
        for f in fields:
            o, c = chi2_balance(f); ones.append(o); chi2s.append(c)
            h1s.append(order1_entropy(f)); lens.append(len(f))
        summary[label] = {
            'n_images': len(fields),
            'avg_field_len': float(np.mean(lens)),
            'avg_ones_frac': float(np.mean(ones)),
            'avg_chi2_balance': float(np.mean(chi2s)),
            'min_chi2_balance': float(np.min(chi2s)),
            'max_chi2_balance': float(np.max(chi2s)),
            'detected_at_p05': int(np.sum(np.array(chi2s) > 3.841)),
            'avg_order1_entropy': float(np.mean(h1s)),
        }
        s = summary[label]
        print("\n[%s] embedded payload field over %d images:" % (label, s['n_images']))
        print("  avg field len      = %.0f bits" % s['avg_field_len'])
        print("  avg ones fraction  = %.4f  (ideal 0.5000)" % s['avg_ones_frac'])
        print("  avg chi2 balance   = %.1f  (crit 3.841 @p=0.05)" % s['avg_chi2_balance'])
        print("  images detected    = %d/%d" % (s['detected_at_p05'], s['n_images']))
        print("  avg order-1 entropy= %.4f bits (ideal 2.0000)" % s['avg_order1_entropy'])

    with open(os.path.join(BASE, 'stego_payload_detectability.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nsaved stego_payload_detectability.json")

if __name__ == '__main__':
    main()
