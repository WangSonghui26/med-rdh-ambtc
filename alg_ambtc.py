import numpy as np
from PIL import Image
import time


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
        # 重构块
        reconstructed_block = np.where(bitmap == 1, h, l)
        # 计算PSNR
        mse = np.mean((block - reconstructed_block) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10(255 ** 2 / mse)
        # 选择PSNR最大的
        if psnr > P_max:
            P_max = psnr
            M_max = bitmap
            h_max = h
            l_max = l

    # 处理所有像素相同的情况（只有1个唯一值，循环未执行）
    if M_max is None:
        M_max = np.ones_like(block, dtype=int)
        h_max = int(distinct_vals[0])
        l_max = int(distinct_vals[0])

    return M_max, h_max, l_max


def ambtc_compress(image, block_size=4, method='proposed'):
    """AMBTC压缩整个图像
    
    Args:
        image (np.ndarray): 输入的图像数组
        block_size (int): 块大小，默认为4
        method (str): 压缩方法，'proposed'表示使用改进的方法，其他值表示使用传统方法
    
    Returns:
        tuple: 包含三个元素的元组:
            - bitmaps (np.ndarray): 所有图像块的位图数组
            - high_labels (np.ndarray): 高位值矩阵，形状为(num_blocks_h, num_blocks_w)
            - low_labels (np.ndarray): 低位值矩阵，形状为(num_blocks_h, num_blocks_w)
    """
    # 获取图像尺寸并计算分块数量
    h, w = image.shape
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    bitmaps = []
    high_labels = []
    low_labels = []

    # 根据指定方法选择压缩函数
    compress_func = proposed_ambtc_compress_block if method == 'proposed' else ambtc_compress_block

    # 遍历每个图像块进行压缩
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # 提取当前块
            block = image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            # 对当前块进行压缩
            bitmap, h_val, l_val = compress_func(block)
            bitmaps.append(bitmap)
            high_labels.append(h_val)
            low_labels.append(l_val)

    # 将结果转换为numpy数组并重新整形
    bitmaps = np.array(bitmaps)
    high_labels = np.array(high_labels).reshape(num_blocks_h, num_blocks_w)
    low_labels = np.array(low_labels).reshape(num_blocks_h, num_blocks_w)

    return bitmaps, high_labels, low_labels

# todo 如果数据不够怎么办？
def jnc_embed_value(value, ref_value, secret_bits):
    """使用JNC和XOR操作嵌入秘密数据到一个值中"""
    s_extra = None
    s1, s2 = secret_bits[:2]
    d = value ^ ref_value

    if d <= 7:
        D = '11'
        d_bits = format(d, '03b')
        s_extra = secret_bits[2:min(7,len(secret_bits))]  # 5 bits 保证不会越界
        stego_bits = s1 + s2 + D + d_bits + s_extra
        # 如果位数不够在后面先补0，保证解密时不会出错。在此处补的0要在解密的时候最后再减去
        while len(stego_bits) < 12:
            stego_bits += '0'
    elif d <= 15:
        D = '10'
        d_bits = format(d - 8, '03b')
        s_extra = secret_bits[2:min(7,len(secret_bits))]  # 5 bits
        stego_bits = s1 + s2 + D + d_bits + s_extra
        # 如果位数不够在后面先补0，保证解密时不会出错。在此处补的0要在解密的时候最后再减去
        while len(stego_bits) < 12:
            stego_bits += '0'
    elif d <= 31:
        D = '01'
        d_bits = format(d - 16, '04b')
        s_extra = secret_bits[2:min(6,len(secret_bits))]  # 4 bits
        stego_bits = s1 + s2 + D + d_bits + s_extra
        # 如果位数不够在后面先补0，保证解密时不会出错。在此处补的0要在解密的时候最后再减去
        while len(stego_bits) < 12:
            stego_bits += '0'
    else:
        D = '00'
        d_bits = format(d, '08b')
        stego_bits = s1 + s2 + D + d_bits

    # if s_extra:
    #     print('secret:',  s1 + s2 + s_extra)
    # else :
    #     print('secret:',  s1 + s2)

    return stego_bits, (len(s_extra) if s_extra else 0) # 应该要返回嵌入的秘密数据长度

# 四位控制位 + 嵌入的数据
def jnc_extract_value(stego_bits, ref_value):
    """从隐写值中提取秘密数据并恢复原始值"""
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
        secret_bits = s1 + s2  + secret_extra
    elif D == '01':
        d_bits = stego_bits[4:8]
        d = int(d_bits, 2) + 16
        secret_extra = stego_bits[8:12]
        secret_bits = s1 + s2  + secret_extra
    else:  # D == '00'
        d_bits = stego_bits[4:12]
        d = int(d_bits, 2)
        secret_bits = s1 + s2

    # if secret_bits:
    #     print('decsecret:',  secret_bits)

    original_value = d ^ ref_value
    return original_value, secret_bits


def _embed_single_table(labels, secret_bits):
    """嵌入秘密数据到单个标签表中（T_h 或 T_l）

    论文 Section 3.2 proposed 嵌入方案：
    1. 非边界值（Algorithm 2）：s1,s2 从秘密数据选取，决定四方向参考值
    2. 边界值：参考值由位置决定（第一行用左邻居，其他用上方邻居），s1,s2 仍从秘密数据选取并前置
    3. [0,0] 保持原始 8 位不变

    嵌入顺序：非边界（从下到上、从右到左）→ 边界列（从下到上）→ 第一行（从右到左）

    Args:
        labels: 标签值矩阵（高标签或低标签）
        secret_bits: 待嵌入的秘密数据位串

    Returns:
        tuple: (stego_labels, secret_bits_remaining, embedded_bits)
    """
    h, w = labels.shape
    embedded_bits = 0
    stego_labels = np.zeros_like(labels, dtype=object)

    # 阶段1: 非边界值（从下到上，从右到左）- Algorithm 2: s1,s2决定参考方向
    for i in range(h - 1, 0, -1):
        for j in range(w - 2, 0, -1):
            if len(secret_bits) <= 0:
                stego_labels[i, j] = format(labels[i, j], '08b')
            else:
                s1, s2 = secret_bits[0], secret_bits[1]
                if s1 == '0' and s2 == '0':
                    ref_value = labels[i, j - 1]       # 左边 (00)
                elif s1 == '0' and s2 == '1':
                    ref_value = labels[i - 1, j - 1]   # 左上角 (01)
                elif s1 == '1' and s2 == '0':
                    ref_value = labels[i - 1, j]        # 正上方 (10)
                else:
                    ref_value = labels[i - 1, j + 1]   # 右上角 (11)
                stego_bits, extra_len = jnc_embed_value(labels[i, j], ref_value, secret_bits)
                stego_labels[i, j] = stego_bits
                embedded_bits += extra_len + 2
                secret_bits = secret_bits[extra_len + 2:]

    # 阶段2: 边界列（从下到上，先最后一列再第一列）- 位置决定参考值
    for i in range(h - 1, 0, -1):
        for j in [w - 1, 0]:
            if len(secret_bits) <= 0:
                stego_labels[i, j] = format(labels[i, j], '08b')
            else:
                ref_value = labels[i - 1, j]  # 非第一行边界：上方的值
                stego_bits, extra_len = jnc_embed_value(labels[i, j], ref_value, secret_bits)
                stego_labels[i, j] = stego_bits
                embedded_bits += extra_len + 2
                secret_bits = secret_bits[extra_len + 2:]

    # 阶段3: 第一行（从右到左，跳过[0,0]）- 位置决定参考值
    for j in range(w - 1, 0, -1):
        if len(secret_bits) <= 0:
            stego_labels[0, j] = format(labels[0, j], '08b')
        else:
            ref_value = labels[0, j - 1]  # 第一行边界：左边的值
            stego_bits, extra_len = jnc_embed_value(labels[0, j], ref_value, secret_bits)
            stego_labels[0, j] = stego_bits
            embedded_bits += extra_len + 2
            secret_bits = secret_bits[extra_len + 2:]

    # [0,0] 保持原始 8 位
    stego_labels[0, 0] = format(labels[0, 0], '08b')

    return stego_labels, secret_bits, embedded_bits


# todo 如果数据不够怎么办？
def embed_data(high_labels, low_labels, secret_data):
    """嵌入秘密数据到高和低标签表中"""
    secret_bits = ''.join(format(ord(c), '08b') for c in secret_data)
    embedded_bits = 0 # 嵌入的秘密数据长度
    h, w = high_labels.shape

    stego_high_labels = np.zeros_like(high_labels, dtype=object)
    stego_low_labels = np.zeros_like(low_labels, dtype=object)

    # 先处理非边界值
    for i in range(h - 1, 0, -1):  # 从下到上
        for j in range(w - 2, 0, -1):  # 从右到左
            if len(secret_bits) <=0 : # 无数据了，直接返回，没有加密数据的标签值就是8位原值，否则应该是12位
                stego_high_labels[i, j] = format(high_labels[i, j], '08b')
                # return stego_high_labels, stego_low_labels, embedded_bits # 继续完成剩余的标签值，只不过没有嵌入数据
            else :
                # 高标签值嵌入
                s1, s2 = secret_bits[0:2] #TODO 不够了咱办？
                if s1 == '0' and s2 == '0':
                    ref_value = high_labels[i, j-1]  # 左边
                elif s1 == '0' and s2 == '1':
                    ref_value = high_labels[i-1, j-1]  # 左上角
                elif s1 == '1' and s2 == '0':
                    ref_value = high_labels[i-1, j]   # 正上方
                else:
                    ref_value = high_labels[i-1, j+1]  # 右上角
                # secret_extra_len 是除了s1,s2外嵌入的秘密数据长度
                stego_bits, secret_extra_len = jnc_embed_value(high_labels[i, j], ref_value, secret_bits)
                stego_high_labels[i, j] = stego_bits
                embedded_bits += secret_extra_len+2 #包含两个控制位s1s2
                secret_bits = secret_bits[secret_extra_len+2:]  # 移除已嵌入的数据位数（包括s1s2）直接移除，这样处理简单点，省去了 secret_index变量的判断

            if len(secret_bits) <=0 : # 无数据了
                stego_low_labels[i, j] = format(low_labels[i, j], '08b')
                # return stego_high_labels, stego_low_labels, embedded_bits
            else :
                # 低标签值嵌入
                s1, s2 = secret_bits[0:2] #TODO 不够了咱办？
                if s1 == '0' and s2 == '0':
                    ref_value = low_labels[i, j-1]  # 左边
                elif s1 == '0' and s2 == '1':
                    ref_value = low_labels[i-1, j-1]  # 左上角
                elif s1 == '1' and s2 == '0':
                    ref_value = low_labels[i-1, j]   # 正上方
                else:
                    ref_value = low_labels[i-1, j+1]  # 右上角

                # secret_extra_len 是除了s1,s2外嵌入的秘密数据长度
                stego_bits, secret_extra_len  = jnc_embed_value(low_labels[i, j], ref_value, secret_bits)
                embedded_bits += secret_extra_len + 2
                stego_low_labels[i, j] = stego_bits
                secret_bits = secret_bits[secret_extra_len + 2:]  # 移除已嵌入的数据位数（包括s1s2）
    # 处理边界值，只有[0,0]是不变的
    # 从下到上处理最后一排和第一排
    for i in range(h - 1, 0, -1):  # i 从 h-1 到 1, 如：127~1，第一行没有处理
        for j in [w - 1, 0]:  # 先处理最后一列，再处理第一列
            if i == 0 and j == 0:
                # 跳过0,0值
                stego_high_labels[0, 0] = format(high_labels[0, 0], '08b')
                stego_low_labels[0, 0] = format(low_labels[0, 0], '08b')
                continue  # 跳过第一个值

            if len(secret_bits) <= 0:  # 无数据了
                stego_high_labels[i, j] = format(high_labels[i, j], '08b')
                # return stego_high_labels, stego_low_labels, embedded_bits
            else :
                # 高标签值嵌入
                if i == 0:  # 第一行
                    ref_value = high_labels[i, j-1]  # 左边的值
                else:  # 不是第一行
                    ref_value = high_labels[i-1, j]  # 上方的值

                stego_bits, secret_extra_len  = jnc_embed_value(high_labels[i, j], ref_value, secret_bits)
                embedded_bits += secret_extra_len + 2
                stego_high_labels[i, j] = stego_bits
                secret_bits = secret_bits[secret_extra_len + 2:]  # 移除已嵌入的数据位数（包括s1s2）

            if len(secret_bits) <=0 : # 无数据了
                stego_low_labels[i, j] = format(low_labels[i, j], '08b')
                # return stego_high_labels, stego_low_labels, embedded_bits
            else :
                # 低标签值嵌入
                if i == 0:  # 第一行
                    ref_value = low_labels[i, j-1]  # 左边的值
                else:  # 不是第一行
                    ref_value = low_labels[i-1, j]  # 上方的值

                stego_bits, secret_extra_len  = jnc_embed_value(low_labels[i, j], ref_value, secret_bits)
                embedded_bits += secret_extra_len + 2
                stego_low_labels[i, j] = stego_bits
                secret_bits = secret_bits[secret_extra_len + 2:]  # 移除已嵌入的数据位数（包括s1s2）

    # 处理边界值
    # 从右到左处理第一行
    for i in range(0,1):  #
        for j in range(w-1, -1,-1):  # 处理第一行，从右到左
            if i == 0 and j == 0:
                # 跳过0,0值
                stego_high_labels[0, 0] = format(high_labels[0, 0], '08b')
                stego_low_labels[0, 0] = format(low_labels[0, 0], '08b')
                continue  # 跳过第一个值

            if len(secret_bits) <= 0:  # 无数据了
                stego_high_labels[i, j] = format(high_labels[i, j], '08b')
                # return stego_high_labels, stego_low_labels, embedded_bits
            else:
                # 高标签值嵌入
                if i == 0:  # 第一行
                    ref_value = high_labels[i, j - 1]  # 左边的值
                else:  # 不是第一行
                    ref_value = high_labels[i - 1, j]  # 上方的值

                stego_bits, secret_extra_len = jnc_embed_value(high_labels[i, j], ref_value, secret_bits)
                embedded_bits += secret_extra_len + 2
                stego_high_labels[i, j] = stego_bits
                secret_bits = secret_bits[secret_extra_len + 2:]  # 移除已嵌入的数据位数（包括s1s2）

            if len(secret_bits) <= 0:  # 无数据了
                stego_low_labels[i, j] = format(low_labels[i, j], '08b')
                # return stego_high_labels, stego_low_labels, embedded_bits
            else:
                # 低标签值嵌入
                if i == 0:  # 第一行
                    ref_value = low_labels[i, j - 1]  # 左边的值
                else:  # 不是第一行
                    ref_value = low_labels[i - 1, j]  # 上方的值

                stego_bits, secret_extra_len = jnc_embed_value(low_labels[i, j], ref_value, secret_bits)
                embedded_bits += secret_extra_len + 2
                stego_low_labels[i, j] = stego_bits
                secret_bits = secret_bits[secret_extra_len + 2:]  # 移除已嵌入的数据位数（包括s1s2）

    return stego_high_labels, stego_low_labels, embedded_bits

#stego_high, stego_low二进制字符串表示的秘密数据
def extract_data(stego_high_labels, stego_low_labels):
    """从隐写标签表中提取秘密数据"""
    secret_bits = ''
    h, w = stego_high_labels.shape

    # 初始化重建的标签表
    reconstructed_high_labels = np.zeros_like(stego_high_labels, dtype=np.int32)
    reconstructed_low_labels = np.zeros_like(stego_low_labels, dtype=np.int32)

    # 直接赋值第一行第一列标签值（未被修改）
    reconstructed_high_labels[0, 0] = int(stego_high_labels[0, 0],2)
    reconstructed_low_labels[0, 0] = int(stego_low_labels[0, 0],2)

    # 先处理边界值
    # 处理第一行，从左到右处理
    print('处理第一行，从左到右处理')
    for i in range(0,1):
        for j in range(0, w):  # 处理第一行，从左到右 # 从左到右处理第一行
            if i == 0 and j == 0:
                continue  # 跳过第一个值

            # 低标签值提取
            if i == 0:  # 第一行
                ref_value = reconstructed_low_labels[i, j-1]  # 左边的值
            else:  # 不是第一行
                ref_value = reconstructed_low_labels[i-1, j]  # 上方的值

            if len(stego_low_labels[i, j]) == 12:
                original_value, extracted_bits = jnc_extract_value(stego_low_labels[i, j], ref_value)
                # secret_bits += extracted_bits[4:]
                secret_bits = extracted_bits + secret_bits  # extracted_bits是包含了s1s2的
                reconstructed_low_labels[i, j] = original_value  # 存储重建的值
            else : # 8位
                reconstructed_low_labels[i, j] = int(stego_low_labels[i, j],2)

            # 高标签值提取
            if i == 0:  # 第一行
                ref_value = reconstructed_high_labels[i, j - 1]  # 左边的值
            else:  # 不是第一行
                ref_value = reconstructed_high_labels[i - 1, j]  # 上方的值

            # 当隐写比特位数等于8时，将二进制字符串转换为整数并返回
            # 参数: stego_bits - 包含隐写数据的二进制字符串
            # 返回值: 元组(int, str) - 第一个元素是转换后的整数值，第二个元素是空字符串
            if len(stego_high_labels[i, j]) == 12:
                original_value, extracted_bits = jnc_extract_value(stego_high_labels[i, j], ref_value)
                # secret_bits += extracted_bits[4:]  # 去掉前4位控制位
                secret_bits = extracted_bits + secret_bits  # extracted_bits是包含了s1s2的
                reconstructed_high_labels[i, j] = original_value  # 存储重建的值
            else:
                reconstructed_high_labels[i, j] = int(stego_high_labels[i, j], 2)  # 存储重建的值
                secret_bits += ''

    print('处理第一列和最后一列，从上到下处理')
    # 先处理边界值
    # 处理第一列和最后一列，从上到下处理
    for i in range(1, h):
        for j in [0,w - 1]:  # 从左到右处理第一行
            if i == 0 and j == 0:
                continue  # 跳过第一个值

            # 低标签值提取
            if i == 0:  # 第一行
                ref_value = reconstructed_low_labels[i, j - 1]  # 左边的值
            else:  # 不是第一行
                ref_value = reconstructed_low_labels[i - 1, j]  # 上方的值

            if len(stego_low_labels[i, j]) == 12:
                original_value, extracted_bits = jnc_extract_value(stego_low_labels[i, j], ref_value)
                secret_bits = extracted_bits + secret_bits  # extracted_bits是包含了s1s2的
                reconstructed_low_labels[i, j] = original_value  # 存储重建的值
            else:  # 8位
                reconstructed_low_labels[i, j] = int(stego_low_labels[i, j], 2)

            # 高标签值提取
            if i == 0:  # 第一行
                ref_value = reconstructed_high_labels[i, j - 1]  # 左边的值
            else:  # 不是第一行
                ref_value = reconstructed_high_labels[i - 1, j]  # 上方的值

            # 当隐写比特位数等于8时，将二进制字符串转换为整数并返回
            # 参数: stego_bits - 包含隐写数据的二进制字符串
            # 返回值: 元组(int, str) - 第一个元素是转换后的整数值，第二个元素是空字符串
            if len(stego_high_labels[i, j]) == 12:
                original_value, extracted_bits = jnc_extract_value(stego_high_labels[i, j], ref_value)
                secret_bits = extracted_bits + secret_bits  # extracted_bits是包含了s1s2的
                reconstructed_high_labels[i, j] = original_value  # 存储重建的值
            else:
                reconstructed_high_labels[i, j] = int(stego_high_labels[i, j], 2)  # 存储重建的值
                secret_bits += ''


    print('再处理非边界值')
    # 再处理非边界值
    for i in range(1, h):
        for j in range(1, w - 1):

            # 倒过来，高低要倒过来
            # 提取低标签值
            s1, s2 = stego_low_labels[i, j][:2]  # 使用嵌入后标签值的前两位
            if s1 == '0' and s2 == '0':
                ref_value = reconstructed_low_labels[i, j-1]  # 左边
            elif s1 == '0' and s2 == '1':
                ref_value = reconstructed_low_labels[i-1, j-1]  # 左上角
            elif s1 == '1' and s2 == '0':
                ref_value = reconstructed_low_labels[i-1, j]   # 正上方
            else:  # s1 == '1' and s2 == '1'
                ref_value = reconstructed_low_labels[i-1, j+1]  # 右上角

            if len(stego_low_labels[i, j]) == 12:
                original_value, extracted_bits = jnc_extract_value(stego_low_labels[i, j], ref_value)
                # secret_bits += extracted_bits[4:]
                secret_bits = extracted_bits + secret_bits  # extracted_bits是包含了s1s2的
                reconstructed_low_labels[i, j] = original_value  # 存储重建的值
            else :
                reconstructed_low_labels[i, j] = int(stego_low_labels[i, j],2)

            # 提取高标签值
            s1, s2 = stego_high_labels[i, j][:2]  # 使用嵌入后标签值的前两位
            if s1 == '0' and s2 == '0':
                ref_value = reconstructed_high_labels[i, j-1]  # 左边
            elif s1 == '0' and s2 == '1':
                ref_value = reconstructed_high_labels[i-1, j-1]  # 左上角
            elif s1 == '1' and s2 == '0':
                ref_value = reconstructed_high_labels[i-1, j]   # 正上方
            else:  # s1 == '1' and s2 == '1'
                ref_value = reconstructed_high_labels[i-1, j+1]  # 右上角

            if len(stego_high_labels[i, j]) == 12:
                original_value, extracted_bits = jnc_extract_value(stego_high_labels[i, j], ref_value)
                # secret_bits += extracted_bits[4:]  # 去掉前4位控制位
                secret_bits = extracted_bits + secret_bits  # extracted_bits是包含了s1s2的
                reconstructed_high_labels[i, j] = original_value  # 存储重建的值
            else :
                reconstructed_high_labels[i, j] = int(stego_high_labels[i, j],2)


    # 将二进制字符串转换回文本
    secret_text = ''
    for i in range(0, len(secret_bits), 8):
        byte = secret_bits[i:i + 8]
        if len(byte) == 8:
            secret_text += chr(int(byte, 2))

    return secret_text,reconstructed_high_labels,reconstructed_low_labels


def proposed_embed_data(high_labels, low_labels, secret_data):
    """Proposed 嵌入方案：先处理完所有 T_h，再处理所有 T_l（论文 Section 3.2）

    Args:
        high_labels: 高标签值矩阵
        low_labels: 低标签值矩阵
        secret_data: 待嵌入的秘密数据字符串

    Returns:
        tuple: (stego_high_labels, stego_low_labels, embedded_bits)
    """
    secret_bits = ''.join(format(ord(c), '08b') for c in secret_data)

    # 先处理所有高标签值 T_h
    stego_high_labels, secret_bits, embedded_h = _embed_single_table(high_labels, secret_bits)

    # 再处理所有低标签值 T_l
    stego_low_labels, secret_bits, embedded_l = _embed_single_table(low_labels, secret_bits)

    return stego_high_labels, stego_low_labels, embedded_h + embedded_l

def _extract_single_table(stego_labels):
    """从单个隐写标签表中提取秘密数据并恢复原始值

    论文 Section 3.3 proposed 提取方案（与嵌入顺序相反）：
    1. 恢复 [0,0]（8位直接转换）
    2. 边界值（从左到右、从上到下）：位置决定参考值
    3. 非边界值（从上到下、从左到右）：s1,s2 决定参考方向（Algorithm 3）

    使用 prepend 策略恢复正确的秘密数据顺序。

    Args:
        stego_labels: 隐写后的标签值矩阵

    Returns:
        tuple: (secret_bits, reconstructed_labels)
    """
    h, w = stego_labels.shape
    secret_bits = ''
    reconstructed = np.zeros_like(stego_labels, dtype=np.int32)

    # 恢复 [0,0]（未被修改，直接转换）
    reconstructed[0, 0] = int(stego_labels[0, 0], 2)

    # 阶段1: 边界值提取（从左到右、从上到下）- 与嵌入顺序相反
    # 第一行（从左到右）
    for j in range(1, w):
        ref_value = reconstructed[0, j - 1]  # 第一行边界：左边的值
        if len(stego_labels[0, j]) == 12:
            original_value, extracted_bits = jnc_extract_value(stego_labels[0, j], ref_value)
            secret_bits = extracted_bits + secret_bits  # prepend
            reconstructed[0, j] = original_value
        else:
            reconstructed[0, j] = int(stego_labels[0, j], 2)

    # 边界列（从上到下，先第一列再最后一列）
    for i in range(1, h):
        for j in [0, w - 1]:
            ref_value = reconstructed[i - 1, j]  # 非第一行边界：上方的值
            if len(stego_labels[i, j]) == 12:
                original_value, extracted_bits = jnc_extract_value(stego_labels[i, j], ref_value)
                secret_bits = extracted_bits + secret_bits  # prepend
                reconstructed[i, j] = original_value
            else:
                reconstructed[i, j] = int(stego_labels[i, j], 2)

    # 阶段2: 非边界值提取（从上到下、从左到右）- 与嵌入顺序相反
    # Algorithm 3: s1,s2 从 stego 值的前两位读取，决定参考方向
    for i in range(1, h):
        for j in range(1, w - 1):
            s1, s2 = stego_labels[i, j][0], stego_labels[i, j][1]
            if s1 == '0' and s2 == '0':
                ref_value = reconstructed[i, j - 1]       # 左边 (00)
            elif s1 == '0' and s2 == '1':
                ref_value = reconstructed[i - 1, j - 1]   # 左上角 (01)
            elif s1 == '1' and s2 == '0':
                ref_value = reconstructed[i - 1, j]        # 正上方 (10)
            else:
                ref_value = reconstructed[i - 1, j + 1]   # 右上角 (11)

            if len(stego_labels[i, j]) == 12:
                original_value, extracted_bits = jnc_extract_value(stego_labels[i, j], ref_value)
                secret_bits = extracted_bits + secret_bits  # prepend
                reconstructed[i, j] = original_value
            else:
                reconstructed[i, j] = int(stego_labels[i, j], 2)

    return secret_bits, reconstructed


def proposed_extract_data(stego_high_labels, stego_low_labels):
    """Proposed 提取方案：先处理所有 T_h，再处理所有 T_l（论文 Section 3.3）

    Args:
        stego_high_labels: 隐写后的高标签值矩阵
        stego_low_labels: 隐写后的低标签值矩阵

    Returns:
        tuple: (secret_text, reconstructed_high_labels, reconstructed_low_labels)
    """
    # 先提取所有高标签值 T_h
    secret_bits_h, reconstructed_high_labels = _extract_single_table(stego_high_labels)

    # 再提取所有低标签值 T_l
    secret_bits_l, reconstructed_low_labels = _extract_single_table(stego_low_labels)

    # 拼接秘密数据：T_h 的数据在前，T_l 的数据在后
    secret_bits = secret_bits_h + secret_bits_l

    # 将二进制字符串转换回文本
    secret_text = ''
    for i in range(0, len(secret_bits), 8):
        byte = secret_bits[i:i + 8]
        if len(byte) == 8:
            secret_text += chr(int(byte, 2))

    return secret_text, reconstructed_high_labels, reconstructed_low_labels


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
            reconstructed[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = block

    return reconstructed


def calculate_psnr(original, reconstructed):
    """计算PSNR"""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255 ** 2 / mse)


# 测试代码
if __name__ == "__main__":
    # 读取图像
    img = Image.open('./pics/pics3/1-Baboon.bmp').convert('L')
    img_array = np.array(img)

    # 设置参数
    block_size = 4
    method = 'proposed'  # 'traditional' 或 'proposed'
    #secret_message = "Hello, this is a secret message!"
    secret_message = "Hello, this is a secret message!"
    # 1. AMBTC压缩
    print("Compressing image...")
    start_time = time.time()
    bitmaps, high_labels, low_labels = ambtc_compress(img_array, block_size, method=method)
    compression_time = time.time() - start_time
    print(f"Compression time: {compression_time:.2f} seconds")

    #打印完整的bitmaps, high_labels, low_labels到 ambtc_compress.txt中
    # with open('ambtc_compress.txt', 'w') as f:
    #     f.write('bitmaps:\n')
    #     for i, bitmap in enumerate(bitmaps):
    #         f.write(f'Block {i}:\n{bitmap}\n')
    #     f.write('\nhigh_labels:\n')
    #     f.write(str(high_labels))
    #     f.write('\nlow_labels:\n')
    #     f.write(str(low_labels))


    # 2. 嵌入秘密数据
    print("Embedding secret data...")
    # stego_high, stego_low二进制字符串表示的秘密数据
    if method == 'proposed':
        stego_high, stego_low, embedded_bits = proposed_embed_data(high_labels, low_labels, secret_message)
    else:
        stego_high, stego_low, embedded_bits = embed_data(high_labels, low_labels, secret_message)
    print(f"Embedded {embedded_bits} bits (method: {method})")

    # 打印完整的stego_high, stego_low, embedded_bits到 embed_data.txt中
    # with open('embed_data.txt', 'w') as f:
    #     f.write('stego_high:\n')
    #     for i in range(stego_high.shape[0]):
    #         for j in range(stego_high.shape[1]):
    #             f.write(f'[{i},{j}]: {stego_high[i, j]}\n')
    #     f.write('\nstego_low:\n')
    #     for i in range(stego_low.shape[0]):
    #         for j in range(stego_low.shape[1]):
    #             f.write(f'[{i},{j}]: {stego_low[i, j]}\n')
    #     f.write(f'\nembedded_bits: {embedded_bits}\n')

    # 3. 提取秘密数据
    print("Extracting secret data...")
    if method == 'proposed':
        extracted_message,reconstructed_high_labels,reconstructed_low_labels = proposed_extract_data(stego_high, stego_low)
    else:
        extracted_message,reconstructed_high_labels,reconstructed_low_labels = extract_data(stego_high, stego_low)
    print(f"Extracted message: {extracted_message}")

    # 4. 重构图像
    print("Reconstructing image...")
    reconstructed_img = reconstruct_image(bitmaps, reconstructed_high_labels, reconstructed_low_labels, block_size)

    # 5. 计算PSNR
    psnr = calculate_psnr(img_array, reconstructed_img)
    print(f"PSNR: {psnr:.2f} dB")

    # 保存重构图像
    Image.fromarray(reconstructed_img).save('reconstructed_image.png')
    print("Reconstructed image saved as 'reconstructed_image.png'")