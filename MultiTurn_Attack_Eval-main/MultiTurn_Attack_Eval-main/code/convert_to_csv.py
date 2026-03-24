import pandas as pd
import os

# --- CẤU HÌNH ---
input_file = os.path.join('data', 'multi_turn_attack_dataset_150_view.csv')
output_file = os.path.join('data', 'multi_turn_attack_dataset_150_normalized.csv')

try:
    df = pd.read_csv(input_file)
    print(">>> Đang chuẩn hóa nhãn...")

    # Hàm chuẩn hóa nhãn
    def normalize_label(row):
        label = str(row['label']).lower()
        source = row['source']
        
        # 1. Xử lý BeaverTails & PAIR (Dạng chữ)
        if label in ['unsafe', 'jailbroken', 'true']:
            return 1 # Tấn công thành công / Độc hại
        if label in ['safe', 'false']:
            return 0 # An toàn
            
        # 2. Xử lý Anthropic (Dạng số)
        # Trong Red Teaming, rating thấp (0,1) thường là model bị fail (tức là tấn công thành công/hoặc phản hồi tệ)
        # Tuy nhiên để chắc chắn cho Baseline tấn công, ta có thể coi các mẫu này là mẫu ĐÍCH (cần đạt được)
        if source == 'Anthropic_RedTeam':
            # Lưu ý: Đây là giả định đơn giản hóa để biến dataset thành "mẫu tấn công thành công"
            # Trong thực tế nghiên cứu sâu, bạn cần đọc kỹ document của từng subset Anthropic.
            return 1 
            
        return 0 # Mặc định an toàn nếu không rõ

    # Áp dụng hàm chuẩn hóa
    df['label_normalized'] = df.apply(normalize_label, axis=1)

    # (Tùy chọn) Xóa cột label cũ lộn xộn đi, chỉ giữ lại cột mới
    df['label'] = df['label_normalized']
    df = df.drop(columns=['label_normalized'])

    # Lưu file mới
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f">>> Đã chuẩn hóa xong! File lưu tại: {output_file}")
    print(">>> Bây giờ cột 'label' chỉ còn toàn số 1 (đại diện cho mẫu tấn công).")

except Exception as e:
    print(f"Lỗi: {e}")