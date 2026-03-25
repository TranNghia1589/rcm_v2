#!/bin/bash

# 🤖 Quick Test Script for Resume Recommendation Chatbot
# Cách sử dụng: bash quick_test.sh

set -e

# Màu output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}  🤖 QUICK CHATBOT TEST SCRIPT${NC}"
echo -e "${BLUE}==========================================${NC}\n"

# Kiểm tra venv
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  venv không tìm thấy${NC}"
    echo "Chạy: python -m venv venv"
    exit 1
fi

# Kích hoạt venv (nếu trên Linux/Mac)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

cd "$(dirname "$0")/.."

echo -e "${GREEN}✓ Venv activated${NC}"
echo -e "${BLUE}Working directory: $(pwd)${NC}\n"

# Hàm print header
print_header() {
    echo -e "\n${BLUE}========== $1 ==========${NC}\n"
}

# Menu chính
while true; do
    echo -e "${YELLOW}Chọn chế độ test:${NC}\n"
    echo "1) 🚀 Test nhanh (dùng dữ liệu sẵn có)"
    echo "2) 📖 Xem hướng dẫn chi tiết"
    echo "3) 🖥️  Chế độ interactive"
    echo "4) 📝 Xem CV samples"
    echo "5) ✅ Test toàn bộ pipeline"
    echo "0) ❌ Thoát\n"

    read -p "Lựa chọn: " choice

    case $choice in
        1)
            print_header "TEST NHANH"

            CV_FILE="data/raw/cv_samples/cv_data_manual.txt"
            EXTRACTED_FILE="data/processed/cv_data_manual_extracted.json"
            GAP_FILE="data/processed/cv_data_manual_gap.json"

            if [ ! -f "$GAP_FILE" ]; then
                echo -e "${YELLOW}📝 Tạo gap analysis trước...${NC}\n"
                python src/cv_processing/extract_cv_info.py \
                    --cv_path "$CV_FILE" \
                    --output_path "$EXTRACTED_FILE"
                python src/matching/gap_analysis.py \
                    --cv_json "$EXTRACTED_FILE" \
                    --output_path "$GAP_FILE"
            fi

            echo -e "\n${GREEN}✓ Sử dụng: $GAP_FILE${NC}\n"

            # Test 3 loại câu hỏi
            echo -e "${BLUE}--- Loại 1: CV Analysis ---${NC}"
            python src/chatbot/chat_router.py \
                --question "CV của tôi phù hợp với vị trí nào nhất?" \
                --gap_result "$GAP_FILE"

            echo -e "\n${BLUE}--- Loại 2: Career Advice ---${NC}"
            python src/chatbot/chat_router.py \
                --question "Nên học gì trong 3 tháng tới?" \
                --gap_result "$GAP_FILE"

            echo -e "\n${BLUE}--- Loại 3: General Question ---${NC}"
            python src/chatbot/chat_router.py \
                --question "Machine Learning là gì?"

            echo -e "\n${GREEN}✓ Test nhanh hoàn thành${NC}"
            ;;

        2)
            print_header "HƯỚNG DẪN CHI TIẾT"
            if [ -f "TESTING_GUIDE.md" ]; then
                less TESTING_GUIDE.md
            else
                echo "File TESTING_GUIDE.md không tìm thấy"
            fi
            ;;

        3)
            print_header "CHẾ ĐỘ INTERACTIVE"
            python test_chatbot_interactive.py
            ;;

        4)
            print_header "CV SAMPLES"
            echo -e "${YELLOW}Sample CVs:${NC}\n"
            ls -lah data/raw/cv_samples/ | grep -E "\.(txt|pdf)$" || echo "Không tìm thấy"
            ;;

        5)
            print_header "TOÀN BỘ PIPELINE"

            echo -e "${BLUE}Bước 1: Extract CV${NC}"
            EXTRACTED="data/processed/pipeline_test_extracted.json"
            python src/cv_processing/extract_cv_info.py \
                --cv_path "data/raw/cv_samples/cv_data_manual.txt" \
                --output_path "$EXTRACTED"
            echo -e "${GREEN}✓ Extracted${NC}\n"

            echo -e "${BLUE}Bước 2: Gap Analysis${NC}"
            GAP="data/processed/pipeline_test_gap.json"
            python src/matching/gap_analysis.py \
                --cv_json "$EXTRACTED" \
                --output_path "$GAP"
            echo -e "${GREEN}✓ Gap Analysis${NC}\n"

            echo -e "${BLUE}Bước 3: Chatbot Test${NC}"
            python src/chatbot/chat_router.py \
                --question "Tôi nên phát triển kỹ năng nào?" \
                --gap_result "$GAP"

            echo -e "\n${GREEN}✓ Pipeline hoàn thành!${NC}"
            ;;

        0)
            echo -e "\n${YELLOW}👋 Tạm biệt!${NC}\n"
            exit 0
            ;;

        *)
            echo -e "${YELLOW}❌ Lựa chọn không hợp lệ${NC}"
            ;;
    esac

    read -p "Nhấn Enter để tiếp tục..."
done
