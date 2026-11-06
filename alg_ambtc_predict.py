"""
改进的AMBTC隐写算法 - 基于预测编码版本
==============================================

算法概述：
    本算法使用多邻居预测编码替代原有的单邻居异或编码。
    通过 MED（Median Edge Detector）预测器，利用左、上、左上三个邻居
    对当前值进行预测，使差值 d 更小，从而嵌入更多秘密数据。
    仅在标签（high_label, low_label）中嵌入数据，不嵌入位图。

============================================================
与原始JNC算法的关键区别
============================================================

    原始JNC算法：
        - s1s2 由秘密数据决定，同时用于选择4个邻居之一
        - d = value XOR selected_neighbor
        - 邻居选择由s1s2"被动"决定，不一定是最优邻居

    预测编码算法：
        - s1s2 仍由秘密数据决定（纯数据承载，不参与预测）
        - 使用 MED 预测器确定性地计算预测值
        - d = value XOR predicted_value
        - MED 预测通常比随机单邻居更接近真实值 → d 更小 → 嵌入更多

============================================================
MED 预测器（Median Edge Detector，JPEG-LS标准使用）
============================================================

    给定三个已知邻居：
        a = 左邻居 (i, j-1)
        b = 上邻居 (i-1, j)
        c = 左上邻居 (i-1, j-1)

    预测规则：
        if c >= max(a, b):
            predicted = min(a, b)   # 水平/垂直边缘
        elif c <= min(a, b):
            predicted = max(a, b)   # 水平/垂直边缘
        else:
            predicted = a + b - c   # 平滑区域

    特点：
        - 自适应边缘检测，在平滑区域和边缘区域都有良好表现
        - 完全确定性，编解码双方可独立计算
        - 广泛用于无损图像压缩（JPEG-LS/LOCO-I）

============================================================
编码结构（与改进版JNC相同）
============================================================

    ┌─────────┬──────────┬───────────────────────────────────────────────┬──────────┐
    │  区间   │ 嵌入量   │              编码结构                          │ 总位数   │
    ├─────────┼──────────┼───────────────────────────────────────────────┼──────────┤
    │ d ≤ 7   │ 7 bits   │ s1s2(2) + D='11'(2) + d(3) + extra(5)       │   12     │
    │ d ≤ 15  │ 7 bits   │ s1s2(2) + D='10'(2) + d-8(3) + extra(5)     │   12     │
    │ d ≤ 31  │ 6 bits   │ s1s2(2) + D='01'(2) + d-16(4) + extra(4)    │   12     │
    │ d ≤ 63  │ 4 bits   │ s1s2(2) + D='00'(2) + sub='1'(1) + d-32(5) + extra(2) │ 12 │
    │ d ≤ 127 │ 2 bits   │ s1s2(2) + D='00'(2) + sub='01'(2) + d-64(6) │   12     │
    │ d > 127 │ 2 bits   │ s1s2(2) + D='00'(2) + sub='00'(2) + d(8)    │   14     │
    └─────────┴──────────┴───────────────────────────────────────────────┴──────────┘

============================================================
数据结构
============================================================

    AMBTC压缩后每个块包含：
    - bitmap: 4×4 = 16位（不嵌入）
    - high_label: 8位
    - low_label: 8位

============================================================
嵌入顺序
============================================================

    标签嵌入（high_labels → low_labels）

    非边界区域：从下到上，从右到左
    第一列：从下到上（上方邻居）
    第一行：从右到左（左方邻居）

============================================================
提取顺序（与嵌入相反）
============================================================

    标签提取（low_labels → high_labels）

    第一行：从左到右
    第一列：从上到下
    非边界区域：从上到下，从左到右

============================================================
可逆性保证
============================================================

    - 预测值由已恢复的邻居确定性计算 → 编解码一致
    - value = d XOR predicted_value → 完全可逆

作者: 改进版 - 预测编码
日期: 2024
"""

import numpy as np
from PIL import Image
import time


# ============================================================
# AMBTC压缩函数
# ============================================================

def ambtc_compress_block(block):
    """传统AMBTC压缩一个块"""
    n = block.shape[0]
    m = np.mean(block).astype(int)
    bitmap = (block >= m).astype(int)
    h = np.mean(block[bitmap == 1]).astype(int) if np.sum(bitmap) > 0 else m
    l = np.mean(block[bitmap == 0]).astype(int) if np.sum(bitmap) < n * n else m
    return bitmap, h, l


def proposed_ambtc_compress_block(block):
    """改进的AMBTC压缩一个块，优化视觉质量"""
    distinct_vals = np.unique(block)
    m = len(distinct_vals)
    P_max = 0
    M_max = None
    h_max = 0
    l_max = 0

    for i in range(m - 1):
        threshold = distinct_vals[i]
        bitmap = (block >= threshold).astype(int)
        h = np.mean(block[bitmap == 1]).astype(int) if np.sum(bitmap) > 0 else threshold
        l = np.mean(block[bitmap == 0]).astype(int) if np.sum(bitmap) < block.size else threshold
        
        reconstructed_block = np.where(bitmap == 1, h, l)
        mse = np.mean((block - reconstructed_block) ** 2)
        
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10(255 ** 2 / mse)
            
        if psnr > P_max:
            P_max = psnr
            M_max = bitmap
            h_max = h
            l_max = l

    return M_max, h_max, l_max


def ambtc_compress(image, block_size=4, method='traditional'):
    """AMBTC压缩整个图像"""
    h, w = image.shape
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    bitmaps = []
    high_labels = []
    low_labels = []

    compress_func = proposed_ambtc_compress_block if method == 'proposed' else ambtc_compress_block

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block = image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            bitmap, h_val, l_val = compress_func(block)
            bitmaps.append(bitmap)
            high_labels.append(h_val)
            low_labels.append(l_val)

    bitmaps = np.array(bitmaps)
    high_labels = np.array(high_labels).reshape(num_blocks_h, num_blocks_w)
    low_labels = np.array(low_labels).reshape(num_blocks_h, num_blocks_w)

    return bitmaps, high_labels, low_labels


# ============================================================
# MED预测器
# ============================================================

def med_predict(a, b, c):
    """MED（Median Edge Detector）预测器
    
    Args:
        a: 左邻居值 (i, j-1)
        b: 上邻居值 (i-1, j)
        c: 左上邻居值 (i-1, j-1)
        
    Returns:
        predicted: 预测值（裁剪到 [0, 255]）
    """
    if c >= max(a, b):
        predicted = min(a, b)
    elif c <= min(a, b):
        predicted = max(a, b)
    else:
        predicted = a + b - c
    
    # 裁剪到有效范围
    return max(0, min(255, predicted))


# ============================================================
# JNC嵌入与提取函数（改进版编码结构）
# ============================================================

def jnc_embed_value(value, ref_value, secret_bits):
    """JNC嵌入 - 使用预测参考值"""
    s_extra = None
    s1, s2 = secret_bits[:2]
    d = value ^ ref_value

    if d <= 7:
        D = '11'
        d_bits = format(d, '03b')
        s_extra = secret_bits[2:min(7, len(secret_bits))]
        stego_bits = s1 + s2 + D + d_bits + s_extra
        while len(stego_bits) < 12:
            stego_bits += '0'
            
    elif d <= 15:
        D = '10'
        d_bits = format(d - 8, '03b')
        s_extra = secret_bits[2:min(7, len(secret_bits))]
        stego_bits = s1 + s2 + D + d_bits + s_extra
        while len(stego_bits) < 12:
            stego_bits += '0'
            
    elif d <= 31:
        D = '01'
        d_bits = format(d - 16, '04b')
        s_extra = secret_bits[2:min(6, len(secret_bits))]
        stego_bits = s1 + s2 + D + d_bits + s_extra
        while len(stego_bits) < 12:
            stego_bits += '0'
            
    elif d <= 63:
        D = '00'
        sub_type = '1'
        d_bits = format(d - 32, '05b')
        s_extra = secret_bits[2:min(4, len(secret_bits))]
        stego_bits = s1 + s2 + D + sub_type + d_bits + s_extra
        while len(stego_bits) < 12:
            stego_bits += '0'
            
    elif d <= 127:
        D = '00'
        sub_type = '01'
        d_bits = format(d - 64, '06b')
        s_extra = ''
        stego_bits = s1 + s2 + D + sub_type + d_bits
        
    else:
        D = '00'
        sub_type = '00'
        d_bits = format(d, '08b')
        s_extra = ''
        stego_bits = s1 + s2 + D + sub_type + d_bits

    return stego_bits, (len(s_extra) if s_extra else 0)


def jnc_extract_value(stego_bits, ref_value):
    """JNC提取 - 使用预测参考值"""
    s1, s2 = stego_bits[:2]
    D = stego_bits[2:4]

    if D == '11':
        d_bits = stego_bits[4:7]
        d = int(d_bits, 2)
        secret_extra = stego_bits[7:12]
        secret_bits = s1 + s2 + secret_extra
        
    elif D == '10':
        d_bits = stego_bits[4:7]
        d = int(d_bits, 2) + 8
        secret_extra = stego_bits[7:12]
        secret_bits = s1 + s2 + secret_extra
        
    elif D == '01':
        d_bits = stego_bits[4:8]
        d = int(d_bits, 2) + 16
        secret_extra = stego_bits[8:12]
        secret_bits = s1 + s2 + secret_extra
        
    else:  # D == '00'
        sub_first = stego_bits[4]
        
        if sub_first == '1':
            d_bits = stego_bits[5:10]
            d = int(d_bits, 2) + 32
            secret_extra = stego_bits[10:12]
            secret_bits = s1 + s2 + secret_extra
            
        else:
            sub_second = stego_bits[5]
            
            if sub_second == '1':
                d_bits = stego_bits[6:12]
                d = int(d_bits, 2) + 64
                secret_bits = s1 + s2
                
            else:
                d_bits = stego_bits[6:14]
                d = int(d_bits, 2)
                secret_bits = s1 + s2

    original_value = d ^ ref_value
    return original_value, secret_bits


# ============================================================
# 基于预测编码的嵌入函数
# ============================================================

def embed_matrix_predict(matrix, secret_bits):
    """基于MED预测编码的矩阵嵌入函数
    
    与原始JNC的区别：
    - 非边界区域使用MED预测器（左、上、左上）计算参考值
    - s1s2仅承载秘密数据，不参与邻居选择
    - 边界区域仍使用相邻单邻居
    
    Args:
        matrix: 8位整数矩阵
        secret_bits: 待嵌入的秘密数据位串
        
    Returns:
        tuple: (stego_matrix, remaining_bits, embedded_bits)
    """
    h, w = matrix.shape
    stego_matrix = np.zeros_like(matrix, dtype=object)
    embedded_bits = 0
    remaining_bits = secret_bits

    # 阶段1: 处理非边界值（从下到上，从右到左）
    # 使用MED预测器：利用左(a)、上(b)、左上(c)三个邻居预测
    for i in range(h - 1, 0, -1):
        for j in range(w - 1, 0, -1):
            if len(remaining_bits) <= 1:
                stego_matrix[i, j] = format(matrix[i, j], '08b')
            else:
                # MED预测
                a = matrix[i, j - 1]       # 左
                b = matrix[i - 1, j]       # 上
                c = matrix[i - 1, j - 1]   # 左上
                ref_value = med_predict(a, b, c)

                stego_bits, secret_extra_len = jnc_embed_value(
                    matrix[i, j], ref_value, remaining_bits)
                stego_matrix[i, j] = stego_bits
                embedded_bits += secret_extra_len + 2
                remaining_bits = remaining_bits[secret_extra_len + 2:]

    # 阶段2: 处理第一列（从下到上），使用上方邻居
    for i in range(h - 1, 0, -1):
        j = 0
        if len(remaining_bits) <= 1:
            stego_matrix[i, j] = format(matrix[i, j], '08b')
        else:
            ref_value = matrix[i - 1, j]  # 上方邻居
            stego_bits, secret_extra_len = jnc_embed_value(
                matrix[i, j], ref_value, remaining_bits)
            stego_matrix[i, j] = stego_bits
            embedded_bits += secret_extra_len + 2
            remaining_bits = remaining_bits[secret_extra_len + 2:]

    # 阶段3: 处理第一行（从右到左），使用左方邻居
    for j in range(w - 1, 0, -1):
        if len(remaining_bits) <= 1:
            stego_matrix[0, j] = format(matrix[0, j], '08b')
        else:
            ref_value = matrix[0, j - 1]  # 左方邻居
            stego_bits, secret_extra_len = jnc_embed_value(
                matrix[0, j], ref_value, remaining_bits)
            stego_matrix[0, j] = stego_bits
            embedded_bits += secret_extra_len + 2
            remaining_bits = remaining_bits[secret_extra_len + 2:]

    # [0,0]位置不嵌入
    stego_matrix[0, 0] = format(matrix[0, 0], '08b')

    return stego_matrix, remaining_bits, embedded_bits


def extract_matrix_predict(stego_matrix):
    """基于MED预测编码的矩阵提取函数
    
    提取顺序与嵌入相反：
    1. [0,0] → 直接还原
    2. 第一行：从左到右，使用左方邻居
    3. 第一列：从上到下，使用上方邻居
    4. 非边界：从上到下、从左到右，使用MED预测
    
    Args:
        stego_matrix: 隐写后的矩阵
        
    Returns:
        tuple: (secret_bits, reconstructed_matrix)
    """
    h, w = stego_matrix.shape
    secret_bits = ''
    reconstructed_matrix = np.zeros_like(stego_matrix, dtype=np.int32)

    # 还原[0,0]
    reconstructed_matrix[0, 0] = int(stego_matrix[0, 0], 2)

    # 阶段1: 处理第一行（从左到右），使用左方邻居
    for j in range(1, w):
        ref_value = reconstructed_matrix[0, j - 1]
        if len(stego_matrix[0, j]) >= 12:
            original_value, extracted_bits = jnc_extract_value(
                stego_matrix[0, j], ref_value)
            secret_bits = extracted_bits + secret_bits
            reconstructed_matrix[0, j] = original_value
        else:
            reconstructed_matrix[0, j] = int(stego_matrix[0, j], 2)

    # 阶段2: 处理第一列（从上到下），使用上方邻居
    for i in range(1, h):
        j = 0
        ref_value = reconstructed_matrix[i - 1, j]
        if len(stego_matrix[i, j]) >= 12:
            original_value, extracted_bits = jnc_extract_value(
                stego_matrix[i, j], ref_value)
            secret_bits = extracted_bits + secret_bits
            reconstructed_matrix[i, j] = original_value
        else:
            reconstructed_matrix[i, j] = int(stego_matrix[i, j], 2)

    # 阶段3: 处理非边界值（从上到下，从左到右），使用MED预测
    for i in range(1, h):
        for j in range(1, w):
            # MED预测：使用已恢复的邻居值
            a = reconstructed_matrix[i, j - 1]       # 左（已恢复）
            b = reconstructed_matrix[i - 1, j]       # 上（已恢复）
            c = reconstructed_matrix[i - 1, j - 1]   # 左上（已恢复）
            ref_value = med_predict(a, b, c)

            if len(stego_matrix[i, j]) >= 12:
                original_value, extracted_bits = jnc_extract_value(
                    stego_matrix[i, j], ref_value)
                secret_bits = extracted_bits + secret_bits
                reconstructed_matrix[i, j] = original_value
            else:
                reconstructed_matrix[i, j] = int(stego_matrix[i, j], 2)

    return secret_bits, reconstructed_matrix


# ============================================================
# 主嵌入和提取函数
# ============================================================

def embed_data(high_labels, low_labels, secret_data):
    """嵌入数据到标签中（基于预测编码）
    
    嵌入顺序：
    1. high_labels
    2. low_labels
    
    Args:
        high_labels: 高标签值矩阵
        low_labels: 低标签值矩阵
        secret_data: 待嵌入的秘密数据字符串
        
    Returns:
        tuple: (stego_high_labels, stego_low_labels, total_embedded_bits)
    """
    secret_bits = ''.join(format(ord(c), '08b') for c in secret_data)
    
    total_embedded = 0
    
    # 1. 嵌入 high_labels
    stego_high_labels, remaining_bits, embedded = embed_matrix_predict(high_labels, secret_bits)
    total_embedded += embedded
    
    # 2. 嵌入 low_labels
    stego_low_labels, remaining_bits, embedded = embed_matrix_predict(low_labels, remaining_bits)
    total_embedded += embedded
    
    return stego_high_labels, stego_low_labels, total_embedded


def extract_data(stego_high_labels, stego_low_labels):
    """从标签中提取数据（基于预测编码）
    
    提取顺序（与嵌入相反）：
    1. low_labels
    2. high_labels
    
    Args:
        stego_high_labels: 隐写后的高标签值矩阵
        stego_low_labels: 隐写后的低标签值矩阵
        
    Returns:
        tuple: (secret_text, reconstructed_high, reconstructed_low)
    """
    all_secret_bits = ''
    
    # 1. 提取 low_labels
    secret_bits, reconstructed_low_labels = extract_matrix_predict(stego_low_labels)
    all_secret_bits = secret_bits + all_secret_bits
    
    # 2. 提取 high_labels
    secret_bits, reconstructed_high_labels = extract_matrix_predict(stego_high_labels)
    all_secret_bits = secret_bits + all_secret_bits
    
    # 转换为文本
    secret_text = ''
    for i in range(0, len(all_secret_bits), 8):
        byte = all_secret_bits[i:i + 8]
        if len(byte) == 8:
            secret_text += chr(int(byte, 2))
    
    return secret_text, reconstructed_high_labels, reconstructed_low_labels


# ============================================================
# 辅助函数
# ============================================================

def reconstruct_image(bitmaps, high_labels, low_labels, block_size=4):
    """从AMBTC数据重构图像"""
    h, w = high_labels.shape
    img_h, img_w = h * block_size, w * block_size
    reconstructed = np.zeros((img_h, img_w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            bitmap = bitmaps[i * w + j]
            h_val = high_labels[i, j]
            l_val = low_labels[i, j]
            block = np.where(bitmap == 1, h_val, l_val).reshape(block_size, block_size)
            reconstructed[i * block_size:(i + 1) * block_size, 
                         j * block_size:(j + 1) * block_size] = block

    return reconstructed


def calculate_psnr(original, reconstructed):
    """计算PSNR"""
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255 ** 2 / mse)


def analyze_d_distribution(matrix, name="matrix"):
    """分析矩阵中的d分布（使用MED预测）"""
    h, w = matrix.shape
    d_values = []
    
    # 非边界区域：MED预测
    for i in range(1, h):
        for j in range(1, w):
            a = matrix[i, j - 1]
            b = matrix[i - 1, j]
            c = matrix[i - 1, j - 1]
            predicted = med_predict(a, b, c)
            d = matrix[i, j] ^ predicted
            d_values.append(d)
    
    # 第一行：左邻居
    for j in range(1, w):
        d = matrix[0, j] ^ matrix[0, j - 1]
        d_values.append(d)
    
    # 第一列：上邻居
    for i in range(1, h):
        d = matrix[i, 0] ^ matrix[i - 1, 0]
        d_values.append(d)
    
    d_ranges = {'d<=7': 0, 'd<=15': 0, 'd<=31': 0,
               'd<=63': 0, 'd<=127': 0, 'd>127': 0}
    
    embed_map = {
        'd<=7': (7, 4), 'd<=15': (7, 4), 'd<=31': (6, 4),
        'd<=63': (4, 4), 'd<=127': (2, 4), 'd>127': (2, 6)
    }
    
    total_embed = 0
    total_storage_increase = 0
    
    for d in d_values:
        if d <= 7:
            key = 'd<=7'
        elif d <= 15:
            key = 'd<=15'
        elif d <= 31:
            key = 'd<=31'
        elif d <= 63:
            key = 'd<=63'
        elif d <= 127:
            key = 'd<=127'
        else:
            key = 'd>127'
        
        d_ranges[key] += 1
        embed, storage = embed_map[key]
        total_embed += embed
        total_storage_increase += storage
    
    return d_ranges, total_embed, total_storage_increase, len(d_values)


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("改进的AMBTC隐写算法 - 基于MED预测编码版本")
    print("=" * 70)
    
    # 读取图像
    img_path = './pics/pics3/1-Baboon.bmp'
    try:
        img = Image.open(img_path).convert('L')
    except FileNotFoundError:
        print(f"图像文件不存在: {img_path}")
        print("使用随机图像进行测试...")
        img_array = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    else:
        img_array = np.array(img)
    
    print(f"图像尺寸: {img_array.shape}")

    # 设置参数
    block_size = 4
    secret_message = "Hello, this is a secret message for testing the AMBTC steganography algorithm with MED predictive coding! " * 10
    print(f"秘密消息长度: {len(secret_message)} 字符 ({len(secret_message) * 8} bits)")
    
    # 1. AMBTC压缩
    print("\n[1] AMBTC压缩...")
    start_time = time.time()
    bitmaps, high_labels, low_labels = ambtc_compress(img_array, block_size, method='traditional')
    compression_time = time.time() - start_time
    print(f"    压缩耗时: {compression_time:.3f} 秒")
    print(f"    标签矩阵尺寸: {high_labels.shape}")
    
    # 2. 分析d分布（MED预测）
    print("\n[2] 分析d值分布（MED预测）...")
    
    total_embed_all = 0
    total_storage_all = 0
    for matrix, name in [(high_labels, 'high_label'), (low_labels, 'low_label')]:
        d_ranges, total_embed, total_storage, count = analyze_d_distribution(matrix, name)
        net = total_embed - total_storage
        total_embed_all += total_embed
        total_storage_all += total_storage
        print(f"\n    {name}:")
        print(f"      样本数: {count}")
        for key, val in d_ranges.items():
            pct = val / count * 100 if count > 0 else 0
            print(f"        {key}: {val} ({pct:.1f}%)")
        print(f"      嵌入量: {total_embed} bits, 存储增加: {total_storage} bits")
        print(f"      净收益: {net:+} bits {'✓' if net > 0 else '✗'}")
    
    print(f"\n    总净收益: {total_embed_all - total_storage_all:+} bits")

    # 3. 嵌入秘密数据
    print("\n[3] 嵌入秘密数据...")
    start_time = time.time()
    stego_high, stego_low, total_embedded = embed_data(
        high_labels, low_labels, secret_message)
    embed_time = time.time() - start_time
    
    print(f"    嵌入耗时: {embed_time:.3f} 秒")
    print(f"    总嵌入: {total_embedded} bits")

    # 4. 提取秘密数据
    print("\n[4] 提取秘密数据...")
    start_time = time.time()
    extracted_message, reconstructed_high, reconstructed_low = extract_data(
        stego_high, stego_low)
    extract_time = time.time() - start_time
    print(f"    提取耗时: {extract_time:.3f} 秒")
    
    if len(extracted_message) > 60:
        print(f"    提取消息: {extracted_message[:60]}...")
    else:
        print(f"    提取消息: {extracted_message}")

    # 5. 验证正确性
    print("\n[5] 验证...")
    extracted_len = len(extracted_message)
    if extracted_len > 0 and secret_message[:extracted_len] == extracted_message:
        print("    ✓ 秘密数据提取正确!")
    else:
        print("    ✗ 秘密数据提取错误!")
        print(f"    原始: {secret_message[:40]}...")
        print(f"    提取: {extracted_message[:40]}...")
    
    high_match = np.array_equal(high_labels, reconstructed_high)
    low_match = np.array_equal(low_labels, reconstructed_low)
    print(f"    高标签值恢复: {'✓ 正确' if high_match else '✗ 错误'}")
    print(f"    低标签值恢复: {'✓ 正确' if low_match else '✗ 错误'}")

    # 6. 重构图像并计算PSNR
    print("\n[6] 重构图像...")
    reconstructed_img = reconstruct_image(bitmaps, reconstructed_high, reconstructed_low, block_size)
    psnr = calculate_psnr(img_array, reconstructed_img)
    print(f"    PSNR: {psnr:.2f} dB")

    # 7. 嵌入率统计
    print("\n[7] 嵌入率统计...")
    total_pixels = img_array.shape[0] * img_array.shape[1]
    bpp = total_embedded / total_pixels
    print(f"    总像素数: {total_pixels}")
    print(f"    总嵌入量: {total_embedded} bits")
    print(f"    嵌入率 (bpp): {bpp:.4f}")
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
