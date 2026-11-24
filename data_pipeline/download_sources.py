"""Download the raw OPUS zh-vi corpora and FLORES-200 into data/raw."""
import os
import sys
import time
import zipfile
import tarfile
import shutil
import urllib.request
import urllib.error
import ssl

# Thiết lập UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thiết lập thư mục gốc dựa trên vị trí của script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
BENCHMARK_DIR = os.path.join(DATA_DIR, "benchmark")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Danh sách dataset cần tải
OPUS_DATASETS = {
    "alt": {
        "url": "https://object.pouta.csc.fi/OPUS-ALT/v20191206/moses/vi-zh.txt.zip",
        "zip_name": "vi-zh.txt.zip",
        "target_dir": os.path.join(RAW_DIR, "alt"),
        "expected_files": ["ALT.vi-zh.vi", "ALT.vi-zh.zh"],
        "approx_size_mb": 2.1
    },
    "ted2020": {
        "url": "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/vi-zh.txt.zip",
        "zip_name": "vi-zh.txt.zip",
        "target_dir": os.path.join(RAW_DIR, "ted2020"),
        "expected_files": ["TED2020.vi-zh.vi", "TED2020.vi-zh.zh"],
        "approx_size_mb": 1.1
    },
    "wikimatrix": {
        "url": "https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/vi-zh.txt.zip",
        "zip_name": "vi-zh.txt.zip",
        "target_dir": os.path.join(RAW_DIR, "wikimatrix"),
        "expected_files": ["WikiMatrix.vi-zh.vi", "WikiMatrix.vi-zh.zh"],
        "approx_size_mb": 10.9
    },
    "ccmatrix": {
        "url": "https://object.pouta.csc.fi/OPUS-CCMatrix/v1/moses/vi-zh.txt.zip",
        "zip_name": "vi-zh.txt.zip",
        "target_dir": os.path.join(RAW_DIR, "ccmatrix"),
        "expected_files": ["CCMatrix.vi-zh.vi", "CCMatrix.vi-zh.zh"],
        "approx_size_mb": 616.4
    }
}

FLORES_TARBALL_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"


def create_directories():
    """Tạo cấu trúc thư mục cần thiết"""
    print("=" * 60)
    print("1. KHỞI TẠO CẤU TRÚC THƯ MỤC")
    print("=" * 60)
    dirs = [
        RAW_DIR,
        os.path.join(RAW_DIR, "flores200"),
        os.path.join(RAW_DIR, "alt"),
        os.path.join(RAW_DIR, "ted2020"),
        os.path.join(RAW_DIR, "wikimatrix"),
        os.path.join(RAW_DIR, "ccmatrix"),
        BENCHMARK_DIR,
        os.path.join(DATA_DIR, "processed"),
        os.path.join(DATA_DIR, "final"),
        LOGS_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f" -> Đã sẵn sàng: {os.path.relpath(d, PROJECT_ROOT)}")
    print()


def download_file_with_progress(url, dest_path):
    """Tải tệp kèm theo thanh hiển thị tiến độ (Progress Bar)"""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    
    start_time = time.time()
    ssl_context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(req, context=ssl_context) as response, open(dest_path, "wb") as out_file:
            total_size = response.getheader("Content-Length")
            if total_size is not None:
                total_size = int(total_size)
            downloaded = 0
            block_size = 1024 * 512  # 512 KB chunks

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                downloaded += len(buffer)
                
                # Hiển thị tiến độ
                elapsed = time.time() - start_time
                speed = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                if total_size:
                    pct = downloaded * 100 / total_size
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r    [{pct:6.2f}%] {mb_downloaded:7.2f}/{mb_total:7.2f} MB ({speed:5.2f} MB/s)")
                else:
                    mb_downloaded = downloaded / (1024 * 1024)
                    sys.stdout.write(f"\r    Đã tải: {mb_downloaded:7.2f} MB ({speed:5.2f} MB/s)")
                sys.stdout.flush()
        print("\n    -> Tải thành công!")
        return True
    except Exception as e:
        print(f"\n    [LỖI] Không thể tải từ {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def extract_zip(zip_path, target_dir):
    """Giải nén tệp zip an toàn"""
    print(f" -> Đang giải nén: {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)
    print(f"    -> Đã giải nén vào: {os.path.relpath(target_dir, PROJECT_ROOT)}")


def download_opus_datasets():
    """Tải và giải nén 4 bộ dữ liệu từ OPUS"""
    print("=" * 60)
    print("2. TẢI VÀ GIẢI NÉN CÁC DATASET HUẤN LUYỆN TỪ OPUS")
    print("=" * 60)

    for name, info in OPUS_DATASETS.items():
        print(f"\n[*] Đang xử lý dataset: {name.upper()} (~{info['approx_size_mb']} MB)")
        target_dir = info["target_dir"]
        zip_path = os.path.join(target_dir, info["zip_name"])
        
        # Kiểm tra xem các file đã được giải nén trước đó chưa
        all_extracted = True
        for ef in info["expected_files"]:
            if not os.path.exists(os.path.join(target_dir, ef)):
                all_extracted = False
                break
                
        if all_extracted:
            print(f"    -> Dữ liệu {name.upper()} đã tồn tại đầy đủ, bỏ qua bước tải.")
            continue

        # Kiểm tra file zip đã tải chưa
        if not os.path.exists(zip_path):
            print(f"    -> Bắt đầu tải từ: {info['url']}")
            success = download_file_with_progress(info["url"], zip_path)
            if not success:
                print(f"    [CẢNH BÁO] Tải {name.upper()} thất bại!")
                continue
        else:
            print(f"    -> Đã có tệp nén: {os.path.basename(zip_path)}")

        # Giải nén
        extract_zip(zip_path, target_dir)


def download_flores200():
    """Tải và trích xuất FLORES-200 cho cặp zho_Hans và vie_Latn"""
    print("=" * 60)
    print("3. TẢI VÀ TRÍCH XUẤT BENCHMARK FLORES-200")
    print("=" * 60)

    flores_raw_dir = os.path.join(RAW_DIR, "flores200")
    tar_path = os.path.join(flores_raw_dir, "flores200_dataset.tar.gz")

    # Danh sách file FLORES-200 đích thực sự cần cho bài toán
    needed_files = {
        "dev/zho_Hans.dev": os.path.join(flores_raw_dir, "dev.zho_Hans"),
        "dev/vie_Latn.dev": os.path.join(flores_raw_dir, "dev.vie_Latn"),
        "devtest/zho_Hans.devtest": os.path.join(flores_raw_dir, "devtest.zho_Hans"),
        "devtest/vie_Latn.devtest": os.path.join(flores_raw_dir, "devtest.vie_Latn")
    }

    # Kiểm tra xem các file đã có chưa
    all_exist = all(os.path.exists(p) for p in needed_files.values())
    if all_exist:
        print(" -> Các tệp raw FLORES-200 đã tồn tại đầy đủ, bỏ qua bước tải.")
    else:
        if os.path.exists(tar_path):
            if os.path.getsize(tar_path) < 1024 * 1024:
                os.remove(tar_path)

        if not os.path.exists(tar_path):
            print(f" -> Đang tải FLORES-200 tarball (~25 MB) từ {FLORES_TARBALL_URL}...")
            success = download_file_with_progress(FLORES_TARBALL_URL, tar_path)
            if not success:
                print(" [LỖI] Không thể tải FLORES-200 tarball!")
                return False

        # Trích xuất có chọn lọc (chỉ giải nén các file zho_Hans và vie_Latn)
        print(" -> Đang trích xuất các nhánh zho_Hans và vie_Latn từ tarball...")
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            for member in members:
                for target_suffix, out_path in needed_files.items():
                    if member.name.endswith(target_suffix):
                        print(f"    -> Giải nén: {member.name} -> {os.path.basename(out_path)}")
                        f = tar.extractfile(member)
                        if f:
                            with open(out_path, "wb") as out_f:
                                out_f.write(f.read())

    # Chuẩn hóa và sao chép sang data/benchmark/ để đóng băng
    print("\n -> Chuẩn hóa và đóng băng vào data/benchmark/ :")
    benchmark_copies = [
        (needed_files["dev/zho_Hans.dev"], os.path.join(BENCHMARK_DIR, "dev.zh")),
        (needed_files["dev/vie_Latn.dev"], os.path.join(BENCHMARK_DIR, "dev.vi")),
        (needed_files["devtest/zho_Hans.devtest"], os.path.join(BENCHMARK_DIR, "test.zh")),
        (needed_files["devtest/vie_Latn.devtest"], os.path.join(BENCHMARK_DIR, "test.vi"))
    ]

    for src, dst in benchmark_copies:
        if os.path.exists(dst):
            print(f"    [ĐÃ CÓ] {os.path.basename(dst)} đã tồn tại và được đóng băng.")
            continue
        if os.path.exists(src):
            shutil.copy2(src, dst)
            # Thiết lập thuộc tính chỉ đọc (Read-only) để bảo vệ benchmark đóng băng
            try:
                os.chmod(dst, 0o444)
            except Exception:
                pass
            print(f"    [ĐÃ KHÓA] {os.path.basename(src)} -> {os.path.relpath(dst, PROJECT_ROOT)}")
        else:
            print(f"    [CẢNH BÁO] Không tìm thấy file nguồn: {src}")

    return True


def count_lines(filepath):
    """Đếm số dòng thực tế trong file văn bản UTF-8"""
    if not os.path.exists(filepath):
        return 0
    count = 0
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            count += 1
    return count


def print_verification_report():
    """Kiểm tra số dòng và tính toàn vẹn của tất cả các file"""
    print("\n" + "=" * 60)
    print("4. BÁO CÁO NGHIỆM THU VÀ KIỂM TRA TÍNH TOÀN VẸN (SANITY CHECK)")
    print("=" * 60)

    # 1. Kiểm tra Benchmark
    print("\n--- A. TẬP BENCHMARK ĐÓNG BĂNG (FLORES-200) ---")
    benchmark_files = [
        ("dev.zh", os.path.join(BENCHMARK_DIR, "dev.zh"), 997),
        ("dev.vi", os.path.join(BENCHMARK_DIR, "dev.vi"), 997),
        ("test.zh", os.path.join(BENCHMARK_DIR, "test.zh"), 1012),
        ("test.vi", os.path.join(BENCHMARK_DIR, "test.vi"), 1012),
    ]
    for label, path, expected in benchmark_files:
        lines = count_lines(path)
        size_kb = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
        status = "[CHÍNH XÁC]" if lines == expected else f"[LỆCH: kỳ vọng {expected}]"
        print(f"  {label:<10}: {lines:>6} dòng | {size_kb:8.2f} KB | {status}")

    # 2. Kiểm tra Dữ liệu huấn luyện
    print("\n--- B. CÁC TẬP DỮ LIỆU HUẤN LUYỆN (RAW OPUS) ---")
    raw_files = [
        ("ALT (vi)", os.path.join(RAW_DIR, "alt", "ALT.vi-zh.vi")),
        ("ALT (zh)", os.path.join(RAW_DIR, "alt", "ALT.vi-zh.zh")),
        ("TED2020 (vi)", os.path.join(RAW_DIR, "ted2020", "TED2020.vi-zh.vi")),
        ("TED2020 (zh)", os.path.join(RAW_DIR, "ted2020", "TED2020.vi-zh.zh")),
        ("WikiMatrix (vi)", os.path.join(RAW_DIR, "wikimatrix", "WikiMatrix.vi-zh.vi")),
        ("WikiMatrix (zh)", os.path.join(RAW_DIR, "wikimatrix", "WikiMatrix.vi-zh.zh")),
        ("CCMatrix (vi)", os.path.join(RAW_DIR, "ccmatrix", "CCMatrix.vi-zh.vi")),
        ("CCMatrix (zh)", os.path.join(RAW_DIR, "ccmatrix", "CCMatrix.vi-zh.zh")),
    ]

    for label, path in raw_files:
        if os.path.exists(path):
            lines = count_lines(path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {label:<18}: {lines:>10,} dòng | {size_mb:8.2f} MB | [SẴN SÀNG]")
        else:
            print(f"  {label:<18}: Chưa tìm thấy file!")

    print("\n" + "=" * 60)
    print("HOÀN TẤT TẢI VÀ THIẾT LẬP DỮ LIỆU!")
    print("=" * 60)


if __name__ == "__main__":
    create_directories()
    download_flores200()
    download_opus_datasets()
    print_verification_report()
