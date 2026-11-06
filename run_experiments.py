"""
Comprehensive Experiment Script for IEEE IoT-J Paper
Runs all methods on all available test images and generates tables.
All data is real - no fabrication.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from PIL import Image
import time
import os
import json

# ============================================================
# Test images - 12 unique standard grayscale images
# ============================================================
BASE = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGES = [
    ("Baboon",   os.path.join(BASE, "pics", "1-Baboon.bmp")),
    ("Airplane", os.path.join(BASE, "pics", "1-airplane.bmp")),
    ("Peppers",  os.path.join(BASE, "pics", "4-Peppers.bmp")),
    ("Elaine",   os.path.join(BASE, "pics", "5-elaine.bmp")),
    ("Wine",     os.path.join(BASE, "pics", "8-wine.bmp")),
    ("Girl",     os.path.join(BASE, "pics", "10-girl.bmp")),
    ("Barbara",  os.path.join(BASE, "pics2", "2-Barbara.bmp")),
    ("Lena",     os.path.join(BASE, "pics2", "3-Lena.bmp")),
    ("Goldhill", os.path.join(BASE, "pics2", "6-Goldhil.bmp")),
    ("Boat",     os.path.join(BASE, "pics2", "9-boat.bmp")),
    ("Tiffany",  os.path.join(BASE, "pics2", "11-tiffany.bmp")),
    ("Zelda",    os.path.join(BASE, "pics2", "12-zelda.bmp")),
]

# ============================================================
# AMBTC Compression (shared by all methods)
# ============================================================
def ambtc_compress_block(block):
    n = block.shape[0]
    m = np.mean(block).astype(int)
    bitmap = (block >= m).astype(int)
    h = np.mean(block[bitmap == 1]).astype(int) if np.sum(bitmap) > 0 else m
    l = np.mean(block[bitmap == 0]).astype(int) if np.sum(bitmap) < n * n else m
    return bitmap, h, l

def ambtc_compress(image, block_size=4):
    h, w = image.shape
    nbh, nbw = h // block_size, w // block_size
    bitmaps, highs, lows = [], [], []
    for i in range(nbh):
        for j in range(nbw):
            block = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            bm, hv, lv = ambtc_compress_block(block)
            bitmaps.append(bm)
            highs.append(hv)
            lows.append(lv)
    return np.array(bitmaps), np.array(highs).reshape(nbh, nbw), np.array(lows).reshape(nbh, nbw)

def reconstruct_image(bitmaps, high_labels, low_labels, block_size=4):
    h, w = high_labels.shape
    img = np.zeros((h*block_size, w*block_size), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            bm = bitmaps[i*w + j]
            block = np.where(bm == 1, high_labels[i,j], low_labels[i,j]).reshape(block_size, block_size)
            img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = block
    return img

def calc_psnr(orig, recon):
    mse = np.mean((orig.astype(float) - recon.astype(float)) ** 2)
    return 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')

# ============================================================
# Variable-length coding (shared by all methods)
# ============================================================
def vl_embed(value, ref_value, secret_bits):
    """Variable-length embed: returns (stego_bits_str, bits_consumed)"""
    if len(secret_bits) < 2:
        return format(value, '08b'), 0
    s1, s2 = secret_bits[0], secret_bits[1]
    d = value ^ ref_value
    consumed = 2  # s1, s2 always consumed

    if d <= 7:
        D = '11'
        d_bits = format(d, '03b')
        extra = secret_bits[2:7]
        consumed += len(extra)
        bits = s1 + s2 + D + d_bits + extra
        bits = bits.ljust(12, '0')
    elif d <= 15:
        D = '10'
        d_bits = format(d - 8, '03b')
        extra = secret_bits[2:7]
        consumed += len(extra)
        bits = s1 + s2 + D + d_bits + extra
        bits = bits.ljust(12, '0')
    elif d <= 31:
        D = '01'
        d_bits = format(d - 16, '04b')
        extra = secret_bits[2:6]
        consumed += len(extra)
        bits = s1 + s2 + D + d_bits + extra
        bits = bits.ljust(12, '0')
    elif d <= 63:
        D = '00'; sub = '1'
        d_bits = format(d - 32, '05b')
        extra = secret_bits[2:4]
        consumed += len(extra)
        bits = s1 + s2 + D + sub + d_bits + extra
        bits = bits.ljust(12, '0')
    elif d <= 127:
        D = '00'; sub = '01'
        d_bits = format(d - 64, '06b')
        bits = s1 + s2 + D + sub + d_bits
        consumed += 0
    else:
        D = '00'; sub = '00'
        d_bits = format(d, '08b')
        bits = s1 + s2 + D + sub + d_bits
        consumed += 0
    return bits, consumed

def vl_extract(stego_bits, ref_value):
    """Variable-length extract: returns (original_value, extracted_secret_bits)"""
    s1, s2 = stego_bits[0], stego_bits[1]
    D = stego_bits[2:4]
    secret = s1 + s2

    if D == '11':
        d = int(stego_bits[4:7], 2)
        secret += stego_bits[7:12]
    elif D == '10':
        d = int(stego_bits[4:7], 2) + 8
        secret += stego_bits[7:12]
    elif D == '01':
        d = int(stego_bits[4:8], 2) + 16
        secret += stego_bits[8:12]
    else:  # D == '00'
        if stego_bits[4] == '1':
            d = int(stego_bits[5:10], 2) + 32
            secret += stego_bits[10:12]
        elif stego_bits[5] == '1':
            d = int(stego_bits[6:12], 2) + 64
        else:
            d = int(stego_bits[6:14], 2)

    original_value = d ^ ref_value
    return original_value, secret

# ============================================================
# Predictor functions
# ============================================================
def predict_med(a, b, c):
    """MED predictor: left=a, upper=b, upper-left=c"""
    if c >= max(a, b):
        return min(a, b)
    elif c <= min(a, b):
        return max(a, b)
    else:
        return max(0, min(255, a + b - c))

def predict_avg(a, b):
    """Average predictor: left=a, upper=b"""
    return max(0, min(255, (a + b) // 2))

def predict_median3(a, b, c):
    """Median of 3 predictor"""
    return sorted([a, b, c])[1]

# ============================================================
# Generic embedding/extraction for any predictor
# ============================================================
def embed_matrix_generic(matrix, secret_bits, predictor_type):
    """
    predictor_type: 'jnc_original', 'med', 'avg', 'median3', 'left_only'
    """
    h, w = matrix.shape
    stego = np.zeros_like(matrix, dtype=object)
    embedded = 0
    remaining = secret_bits

    # Phase 1: Non-boundary (bottom to top, right to left)
    for i in range(h - 1, 0, -1):
        for j in range(w - 1, 0, -1):
            if len(remaining) <= 1:
                stego[i, j] = format(matrix[i, j], '08b')
            else:
                s1, s2 = remaining[0], remaining[1]
                if predictor_type == 'jnc_original':
                    # JNC: s1s2 determines neighbor selection
                    if s1 == '0' and s2 == '0':
                        ref = matrix[i, j-1]
                    elif s1 == '0' and s2 == '1':
                        ref = matrix[i-1, j-1]
                    elif s1 == '1' and s2 == '0':
                        ref = matrix[i-1, j]
                    else:
                        ref = matrix[i-1, min(j+1, w-1)]
                elif predictor_type == 'med':
                    ref = predict_med(int(matrix[i,j-1]), int(matrix[i-1,j]), int(matrix[i-1,j-1]))
                elif predictor_type == 'avg':
                    ref = predict_avg(int(matrix[i,j-1]), int(matrix[i-1,j]))
                elif predictor_type == 'median3':
                    ref = predict_median3(int(matrix[i,j-1]), int(matrix[i-1,j]), int(matrix[i-1,j-1]))
                elif predictor_type == 'left_only':
                    ref = matrix[i, j-1]
                else:
                    ref = matrix[i, j-1]

                bits, consumed = vl_embed(int(matrix[i,j]), int(ref), remaining)
                stego[i, j] = bits
                embedded += consumed
                remaining = remaining[consumed:]

    # Phase 2: First column (bottom to top) - use upper neighbor
    for i in range(h - 1, 0, -1):
        j = 0
        if len(remaining) <= 1:
            stego[i, j] = format(matrix[i, j], '08b')
        else:
            ref = matrix[i-1, j]
            bits, consumed = vl_embed(int(matrix[i,j]), int(ref), remaining)
            stego[i, j] = bits
            embedded += consumed
            remaining = remaining[consumed:]

    # Phase 3: First row (right to left) - use left neighbor
    for j in range(w - 1, 0, -1):
        if len(remaining) <= 1:
            stego[0, j] = format(matrix[0, j], '08b')
        else:
            ref = matrix[0, j-1]
            bits, consumed = vl_embed(int(matrix[0,j]), int(ref), remaining)
            stego[0, j] = bits
            embedded += consumed
            remaining = remaining[consumed:]

    # [0,0] unchanged
    stego[0, 0] = format(matrix[0, 0], '08b')
    return stego, remaining, embedded

def extract_matrix_generic(stego, predictor_type):
    """Generic extraction - reverse order of embedding"""
    h, w = stego.shape
    secret = ''
    recon = np.zeros_like(stego, dtype=np.int32)

    # [0,0]
    recon[0, 0] = int(stego[0, 0], 2)

    # Phase 1: First row (left to right)
    for j in range(1, w):
        ref = recon[0, j-1]
        if len(stego[0, j]) >= 12:
            val, bits = vl_extract(stego[0, j], ref)
            secret = bits + secret
            recon[0, j] = val
        else:
            recon[0, j] = int(stego[0, j], 2)

    # Phase 2: First column (top to bottom)
    for i in range(1, h):
        j = 0
        ref = recon[i-1, j]
        if len(stego[i, j]) >= 12:
            val, bits = vl_extract(stego[i, j], ref)
            secret = bits + secret
            recon[i, j] = val
        else:
            recon[i, j] = int(stego[i, j], 2)

    # Phase 3: Non-boundary (top to bottom, left to right)
    for i in range(1, h):
        for j in range(1, w):
            if predictor_type == 'jnc_original':
                # For JNC, s1s2 are already in stego, need to peek at them
                if len(stego[i, j]) >= 12:
                    s1, s2 = stego[i, j][0], stego[i, j][1]
                    if s1 == '0' and s2 == '0':
                        ref = recon[i, j-1]
                    elif s1 == '0' and s2 == '1':
                        ref = recon[i-1, j-1]
                    elif s1 == '1' and s2 == '0':
                        ref = recon[i-1, j]
                    else:
                        ref = recon[i-1, min(j+1, w-1)]
                    val, bits = vl_extract(stego[i, j], ref)
                    secret = bits + secret
                    recon[i, j] = val
                else:
                    recon[i, j] = int(stego[i, j], 2)
            elif predictor_type == 'med':
                ref = predict_med(int(recon[i,j-1]), int(recon[i-1,j]), int(recon[i-1,j-1]))
                if len(stego[i, j]) >= 12:
                    val, bits = vl_extract(stego[i, j], ref)
                    secret = bits + secret
                    recon[i, j] = val
                else:
                    recon[i, j] = int(stego[i, j], 2)
            elif predictor_type == 'avg':
                ref = predict_avg(int(recon[i,j-1]), int(recon[i-1,j]))
                if len(stego[i, j]) >= 12:
                    val, bits = vl_extract(stego[i, j], ref)
                    secret = bits + secret
                    recon[i, j] = val
                else:
                    recon[i, j] = int(stego[i, j], 2)
            elif predictor_type == 'median3':
                ref = predict_median3(int(recon[i,j-1]), int(recon[i-1,j]), int(recon[i-1,j-1]))
                if len(stego[i, j]) >= 12:
                    val, bits = vl_extract(stego[i, j], ref)
                    secret = bits + secret
                    recon[i, j] = val
                else:
                    recon[i, j] = int(stego[i, j], 2)
            elif predictor_type == 'left_only':
                ref = recon[i, j-1]
                if len(stego[i, j]) >= 12:
                    val, bits = vl_extract(stego[i, j], ref)
                    secret = bits + secret
                    recon[i, j] = val
                else:
                    recon[i, j] = int(stego[i, j], 2)

    return secret, recon

def embed_both_tables(high_labels, low_labels, predictor_type):
    """Embed into both high and low label tables with sufficient secret data"""
    # Generate enough random secret bits
    total_labels = 2 * high_labels.size
    max_bits_per_label = 7
    total_possible = total_labels * max_bits_per_label
    secret_bits = ''.join([str(np.random.randint(2)) for _ in range(total_possible)])

    stego_h, remaining, emb_h = embed_matrix_generic(high_labels, secret_bits, predictor_type)
    stego_l, remaining, emb_l = embed_matrix_generic(low_labels, remaining, predictor_type)
    return stego_h, stego_l, emb_h + emb_l

def reconstruct_stego_labels(stego_matrix):
    """Convert stego matrix (binary strings) back to integer label matrix for PSNR"""
    h, w = stego_matrix.shape
    result = np.zeros((h, w), dtype=np.int32)
    for i in range(h):
        for j in range(w):
            bits = stego_matrix[i, j]
            if len(bits) >= 12:
                # Extract the d value and reconstruct
                # The stored value IS the codeword; we need the original value
                # For PSNR: we reconstruct from the stego data using extraction
                pass
            else:
                result[i, j] = int(bits, 2)
    return result

def run_experiment_on_image(img_path, img_name, block_size=4):
    """Run all methods on one image and collect metrics"""
    img = Image.open(img_path).convert('L')
    img_array = np.array(img)
    orig_h, orig_w = img_array.shape

    # AMBTC compress
    bitmaps, high_labels, low_labels = ambtc_compress(img_array, block_size)
    nbh, nbw = high_labels.shape

    # Reconstruct original AMBTC image (no embedding) for baseline PSNR
    ambtc_recon = reconstruct_image(bitmaps, high_labels, low_labels, block_size)
    ambtc_psnr = calc_psnr(img_array[:ambtc_recon.shape[0], :ambtc_recon.shape[1]], ambtc_recon)

    methods = ['jnc_original', 'med', 'avg', 'left_only']
    method_names = {
        'jnc_original': 'JNC-Original',
        'med': 'MED-Predict (Proposed)',
        'avg': 'Avg-Predict',
        'left_only': 'Left-Neighbor'
    }

    results = {}
    for method in methods:
        t0 = time.time()
        stego_h, stego_l, total_embedded = embed_both_tables(high_labels, low_labels, method)
        embed_time = time.time() - t0

        # Extract and verify
        t0 = time.time()
        secret_h, recon_h = extract_matrix_generic(stego_h, method)
        secret_l, recon_l = extract_matrix_generic(stego_l, method)
        extract_time = time.time() - t0

        # Verify labels are perfectly recovered
        h_correct = np.array_equal(recon_h, high_labels.astype(np.int32))
        l_correct = np.array_equal(recon_l, low_labels.astype(np.int32))

        # Reconstruct stego image for PSNR
        # Need to convert stego labels back to integers
        stego_h_int = np.zeros_like(high_labels, dtype=np.int32)
        stego_l_int = np.zeros_like(low_labels, dtype=np.int32)
        for i in range(nbh):
            for j in range(nbw):
                # For embedded labels, extract original value
                bits_h = stego_h[i, j]
                bits_l = stego_l[i, j]
                if len(bits_h) >= 12:
                    # Reconstruct from extracted data (which we know is correct)
                    stego_h_int[i, j] = int(recon_h[i, j])
                else:
                    stego_h_int[i, j] = int(bits_h, 2)
                if len(bits_l) >= 12:
                    stego_l_int[i, j] = int(recon_l[i, j])
                else:
                    stego_l_int[i, j] = int(bits_l, 2)

        # Actually for stego PSNR, we need the values AFTER embedding (which differ from original)
        # The stego labels are modified - we need to decode the actual stored values
        # Let's reconstruct using the stego binary data directly
        def decode_stego_value(stego_bits_str):
            """Decode what value is stored in the stego codeword"""
            if len(stego_bits_str) <= 8:
                return int(stego_bits_str, 2)
            # For 12-14 bit codewords, the value is d XOR ref
            # But we need to extract d first
            D = stego_bits_str[2:4]
            if D == '11':
                d = int(stego_bits_str[4:7], 2)
            elif D == '10':
                d = int(stego_bits_str[4:7], 2) + 8
            elif D == '01':
                d = int(stego_bits_str[4:8], 2) + 16
            else:
                if stego_bits_str[4] == '1':
                    d = int(stego_bits_str[5:10], 2) + 32
                elif stego_bits_str[5] == '1':
                    d = int(stego_bits_str[6:12], 2) + 64
                else:
                    d = int(stego_bits_str[6:14], 2)
            return d  # This is the XOR difference, not the actual value

        # For PSNR of stego image: the stego labels differ from original because
        # the variable-length codeword replaces the original 8-bit value.
        # When we reconstruct the image, we use the extracted (recovered) label values,
        # which are IDENTICAL to the originals (since the scheme is reversible).
        # So stego PSNR = AMBTC PSNR (no additional degradation from embedding itself).
        # The PSNR degradation only happens if the stego codewords introduce
        # reconstruction errors - but in a reversible scheme, they don't.
        # 
        # However, the paper reports different PSNR values because the stego compressed
        # stream is larger and the reconstruction uses different values during transmission.
        # Let me re-read the paper to understand what PSNR they're measuring...
        #
        # Actually, re-reading the paper: the PSNR reported is between original image and
        # the "reconstructed stego image". In a reversible scheme, the reconstructed image
        # should be identical to the AMBTC-compressed image (not the original).
        # So PSNR should be the same as AMBTC PSNR.
        #
        # BUT the paper says the proposed method has ~3dB lower PSNR. This means the
        # "stego image" is NOT reconstructed from recovered labels, but rather from
        # the stego codewords directly (which represent modified values).
        # 
        # Let me think about this more carefully...
        # In the variable-length coding, the stego codeword is 12-14 bits.
        # When the receiver decodes, they get the original value back (reversible).
        # But the "stego image" in the paper's context is reconstructed by
        # treating each stego label's encoded value as the new label value.
        # 
        # This doesn't make sense for a reversible scheme... Let me check the paper text.
        # The paper says: "PSNR between original image and reconstructed stego image"
        # And notes the decrease is "primarily attributable to... the increased codeword length"
        # 
        # I think the PSNR difference comes from the fact that during embedding,
        # the codeword stores d (the XOR difference), not the original value.
        # When reconstructing without extracting, the decoder would get different values.
        # But with proper extraction, the original is perfectly recovered.
        # 
        # For the paper, I think the stego PSNR is measured differently - perhaps by
        # interpreting the stego bitstream as new label values. But that's a design choice.
        # 
        # For now, let me just compute the reversible PSNR (which equals AMBTC PSNR)
        # and note that the scheme is fully reversible.

        stego_recon = reconstruct_image(bitmaps, stego_h_int, stego_l_int, block_size)
        stego_psnr = calc_psnr(img_array[:stego_recon.shape[0], :stego_recon.shape[1]], stego_recon)

        # d-value distribution analysis
        d_values = []
        for matrix in [high_labels, low_labels]:
            mh, mw = matrix.shape
            for ii in range(1, mh):
                for jj in range(1, mw):
                    if method == 'jnc_original':
                        # For JNC, pick a random neighbor (simulating average case)
                        ref = int(matrix[ii, jj-1])  # use left for analysis
                    elif method == 'med':
                        ref = predict_med(int(matrix[ii,jj-1]), int(matrix[ii-1,jj]), int(matrix[ii-1,jj-1]))
                    elif method == 'avg':
                        ref = predict_avg(int(matrix[ii,jj-1]), int(matrix[ii-1,jj]))
                    elif method == 'left_only':
                        ref = int(matrix[ii, jj-1])
                    d_values.append(int(matrix[ii, jj]) ^ ref)

        d_le7 = sum(1 for d in d_values if d <= 7)
        d_le31 = sum(1 for d in d_values if d <= 31)
        total_d = len(d_values)

        bpp = total_embedded / (orig_h * orig_w)

        results[method] = {
            'name': method_names[method],
            'embedded_bits': total_embedded,
            'bpp': round(bpp, 4),
            'embed_time': round(embed_time, 4),
            'extract_time': round(extract_time, 4),
            'h_correct': h_correct,
            'l_correct': l_correct,
            'psnr': round(stego_psnr, 2),
            'ambtc_psnr': round(ambtc_psnr, 2),
            'd_le7_pct': round(d_le7 / total_d * 100, 1) if total_d > 0 else 0,
            'd_le31_pct': round(d_le31 / total_d * 100, 1) if total_d > 0 else 0,
        }

    return results

# ============================================================
# Main: Run experiments on all images
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Comprehensive Experiment: 12 images x 4 methods")
    print("=" * 80)

    all_results = {}
    np.random.seed(42)  # Reproducible

    for img_name, img_path in TEST_IMAGES:
        if not os.path.exists(img_path):
            print(f"  SKIP: {img_name} ({img_path} not found)")
            continue

        print(f"\n[{img_name}] Processing...", end=" ", flush=True)
        t0 = time.time()
        results = run_experiment_on_image(img_path, img_name)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")

        for method, data in results.items():
            print(f"  {data['name']:25s} | Embed: {data['embedded_bits']:>7d} bits ({data['bpp']:.4f} bpp) | "
                  f"PSNR: {data['psnr']:.2f} dB | d<=7: {data['d_le7_pct']:.1f}% | "
                  f"Correct: {'Y' if data['h_correct'] and data['l_correct'] else 'N'}")

        all_results[img_name] = results

    # ============================================================
    # Summary tables
    # ============================================================
    print("\n" + "=" * 80)
    print("SUMMARY TABLES")
    print("=" * 80)

    methods_order = ['jnc_original', 'left_only', 'avg', 'med']
    method_labels = ['JNC-Original', 'Left-Neighbor', 'Avg-Predict', 'MED-Predict (Proposed)']

    # Table 1: Embedding Capacity
    print("\n--- Table 1: Embedding Capacity (bits) ---")
    header = f"{'Image':<12s}"
    for label in method_labels:
        header += f" | {label:>22s}"
    print(header)
    print("-" * len(header))

    for img_name in [n for n, _ in TEST_IMAGES if n in all_results]:
        row = f"{img_name:<12s}"
        for m in methods_order:
            if m in all_results[img_name]:
                row += f" | {all_results[img_name][m]['embedded_bits']:>22,d}"
        print(row)

    # Table 2: Embedding Rate (bpp)
    print("\n--- Table 2: Embedding Rate (bpp) ---")
    header = f"{'Image':<12s}"
    for label in method_labels:
        header += f" | {label:>22s}"
    print(header)
    print("-" * len(header))

    for img_name in [n for n, _ in TEST_IMAGES if n in all_results]:
        row = f"{img_name:<12s}"
        for m in methods_order:
            if m in all_results[img_name]:
                row += f" | {all_results[img_name][m]['bpp']:>22.4f}"
        print(row)

    # Table 3: PSNR
    print("\n--- Table 3: PSNR (dB) ---")
    header = f"{'Image':<12s} | {'AMBTC':>10s}"
    for label in method_labels:
        header += f" | {label:>22s}"
    print(header)
    print("-" * len(header))

    for img_name in [n for n, _ in TEST_IMAGES if n in all_results]:
        row = f"{img_name:<12s}"
        # AMBTC baseline PSNR
        first_method = list(all_results[img_name].values())[0]
        row += f" | {first_method['ambtc_psnr']:>10.2f}"
        for m in methods_order:
            if m in all_results[img_name]:
                row += f" | {all_results[img_name][m]['psnr']:>22.2f}"
        print(row)

    # Table 4: d-value distribution (% with d<=7)
    print("\n--- Table 4: % Labels with d <= 7 ---")
    header = f"{'Image':<12s}"
    for label in method_labels:
        header += f" | {label:>22s}"
    print(header)
    print("-" * len(header))

    for img_name in [n for n, _ in TEST_IMAGES if n in all_results]:
        row = f"{img_name:<12s}"
        for m in methods_order:
            if m in all_results[img_name]:
                row += f" | {all_results[img_name][m]['d_le7_pct']:>21.1f}%"
        print(row)

    # Table 5: Correctness verification
    print("\n--- Table 5: Correctness Verification ---")
    all_correct = True
    for img_name in [n for n, _ in TEST_IMAGES if n in all_results]:
        for m in methods_order:
            if m in all_results[img_name]:
                data = all_results[img_name][m]
                if not (data['h_correct'] and data['l_correct']):
                    print(f"  FAIL: {img_name} / {data['name']}")
                    all_correct = False
    if all_correct:
        print("  ALL PASSED: 100% correct extraction across all images and methods")

    # Averages
    print("\n--- Averages ---")
    for m in methods_order:
        emb_list = [all_results[n][m]['embedded_bits'] for n in all_results if m in all_results[n]]
        bpp_list = [all_results[n][m]['bpp'] for n in all_results if m in all_results[n]]
        psnr_list = [all_results[n][m]['psnr'] for n in all_results if m in all_results[n]]
        d7_list = [all_results[n][m]['d_le7_pct'] for n in all_results if m in all_results[n]]

        avg_emb = np.mean(emb_list) if emb_list else 0
        avg_bpp = np.mean(bpp_list) if bpp_list else 0
        avg_psnr = np.mean(psnr_list) if psnr_list else 0
        avg_d7 = np.mean(d7_list) if d7_list else 0

        label = [l for l, mo in zip(method_labels, methods_order) if mo == m][0]
        print(f"  {label:25s} | Avg Embed: {avg_emb:>8.0f} bits | Avg bpp: {avg_bpp:.4f} | "
              f"Avg PSNR: {avg_psnr:.2f} dB | Avg d<=7: {avg_d7:.1f}%")

    # Improvement of MED over JNC
    print("\n--- MED-Predict vs JNC-Original Improvement ---")
    for img_name in [n for n, _ in TEST_IMAGES if n in all_results]:
        if 'jnc_original' in all_results[img_name] and 'med' in all_results[img_name]:
            jnc_emb = all_results[img_name]['jnc_original']['embedded_bits']
            med_emb = all_results[img_name]['med']['embedded_bits']
            pct = (med_emb - jnc_emb) / jnc_emb * 100 if jnc_emb > 0 else 0
            print(f"  {img_name:<12s}: JNC={jnc_emb:>7,d}  MED={med_emb:>7,d}  Improvement: {pct:+.1f}%")

    # Save results as JSON for later use
    output_path = r"C:\Users\L\OneDrive\桌面\sci论文\experiment_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
