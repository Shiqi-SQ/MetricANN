import os
from PIL import Image

DATA_DIR = r'dataset'

def validate_dataset(data_dir):
    issues = []
    for cls in sorted(os.listdir(data_dir)):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        imgs = [f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"{cls}: {len(imgs)} 张图片")
        for fn in imgs:
            path = os.path.join(cls_dir, fn)
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception as e:
                issues.append((cls, fn, str(e)))
    if issues:
        print("\n存在以下损坏或无法打开的文件：")
        for cls, fn, err in issues:
            print(f"  {cls}/{fn} → {err}")
    else:
        print("\n所有图片均已验证通过！")

if __name__ == '__main__':
    validate_dataset(DATA_DIR)
