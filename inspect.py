"""
라벨링이 잘못되었거나 사용불가한 이미지 이동
1-3 키로 라벨 폴더 선택
d 키로 스킵
f 키로 사용불가 처리
백스페이스키로 검수 처리
ESC: 프로그램 종료

파트명.xlsx: 라벨링이 잘못되었거나 사용불가한 이미지에 대한 정보가 제거된 엑셀
파트명_백업.xlsx: 원본 엑셀

"""

import os
import cv2
import shutil
import pandas as pd
import sys

# 1. 대상 폴더 지정
base_dir = "후드"
label_folders = ["bad", "good", "Y"]
dest_dir = "../검수0717/후드"

# 엑셀 데이터 미리 읽어두기
excel_path = os.path.join(base_dir, "후드.xlsx")
df = pd.read_excel(excel_path)

# 라벨 폴더 선택
print("처리할 라벨 폴더를 선택하세요:")
print("1. bad")
print("2. good") 
print("3. Y")

while True:
    try:
        choice = input("선택 (1-3): ").strip()
        if choice == "1":
            selected_folders = ["bad"]
            break
        elif choice == "2":
            selected_folders = ["good"]
            break
        elif choice == "3":
            selected_folders = ["Y"]
            break
        else:
            print("1-3 중에서 선택해주세요.")
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
        sys.exit(0)

print(f"선택된 폴더: {selected_folders}")

# 2. 이미지 리스트업
image_paths = []
for label in selected_folders:
    folder = os.path.join(base_dir, label)
    if os.path.exists(folder):
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append((label, os.path.join(folder, fname)))

# 처리된 이미지들을 추적하기 위한 리스트들
processed_good = []
processed_bad = []
processed_Y = []
processed_unusable = []

# 대응되는 이미지를 찾는 함수
def find_corresponding_images(df, current_row):
    """동일한 파트명, 차량모델, 차량코드, 차량번호를 가진 이미지들을 찾습니다."""
    part = current_row['파트명']
    model = current_row['차량모델']
    code = current_row['차량코드']
    number = current_row['차량번호']
    
    corresponding = df[
        (df['파트명'] == part) &
        (df['차량모델'] == model) &
        (df['차량코드'] == code) &
        (df['차량번호'] == number)
    ]
    
    return corresponding

# 대응되는 이미지들을 처리하는 함수
def process_corresponding_images(df, current_row, selected_label, base_dir, dest_dir, is_unusable=False):
    """대응되는 이미지들을 찾아서 해당 폴더로 이동시킵니다."""
    corresponding = find_corresponding_images(df, current_row)
    processed_bad_images = []
    processed_Y_images = []
    
    for _, row in corresponding.iterrows():
        filename = row['이미지파일명']
        part_name = row['병합파트명']
        
        # 각 라벨 폴더에서 해당 파일 찾기
        for folder_label in ["bad", "Y"]:
            folder_path = os.path.join(base_dir, folder_label)
            if os.path.exists(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.exists(file_path):
                    # 사용불가 처리인 경우
                    if is_unusable:
                        dest_folder = os.path.join(dest_dir, f"{part_name}_사용불가")
                        # 사용불가로 처리되는 이미지는 bad, Y 구분 없이 모두 사용불가 리스트에 추가
                        if folder_label == "bad":
                            processed_bad_images.append(row.to_dict())
                        elif folder_label == "Y":
                            processed_Y_images.append(row.to_dict())
                    else:
                        # 일반 처리인 경우
                        if selected_label == "bad":
                            # bad를 선택했으면 대응되는 Y는 Y불량검수로
                            dest_folder = os.path.join(dest_dir, f"{part_name}_Y불량검수")
                            processed_Y_images.append(row.to_dict())
                        elif selected_label == "Y":
                            # Y를 선택했으면 대응되는 bad는 불량검수로
                            dest_folder = os.path.join(dest_dir, f"{part_name}_불량검수")
                            processed_bad_images.append(row.to_dict())
                    
                    os.makedirs(dest_folder, exist_ok=True)
                    
                    # 파일 이동
                    dest_path = os.path.join(dest_folder, filename)
                    shutil.move(file_path, dest_path)
                    print(f"대응 이미지 이동: {file_path} -> {dest_path}")
                    break
    
    return processed_bad_images, processed_Y_images


for label, img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지 로드 실패: {img_path}")
        continue
    # 이미지에 폴더 정보와 파일명 영어로 표시
    display_img = img.copy()
    info_text = f"Folder: {label} | File: {os.path.basename(img_path)}"
    cv2.putText(display_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow("Labeling", display_img)
    key = cv2.waitKey(0)
    print(f"Pressed key code: {key}", flush=True)
    if key in [ord('d'), 71]:
        print(f"Skipped: {img_path}", flush=True)
        cv2.destroyAllWindows()
        continue
    elif key in [8, 127]:  # 8: Windows, 127: macOS/Linux
        # 엑셀에서 병합파트명 가져오기
        fname = os.path.basename(img_path)
        row = df[df['이미지파일명'] == fname]
        if not row.empty:
            part_name = row.iloc[0]['병합파트명']
            current_row = row.iloc[0]
        else:
            part_name = "unknown"
            current_row = None
            
        if label == "good":
            dest_folder = os.path.join(dest_dir, f"{part_name}_양품검수")
            os.makedirs(dest_folder, exist_ok=True)
            dest_path = os.path.join(dest_folder, os.path.basename(img_path))
            shutil.move(img_path, dest_path)
            print(f"Moved (good): {img_path} -> {dest_path}", flush=True)
            
            # 양품검수 처리된 이미지 정보 저장
            if current_row is not None:
                processed_good.append(current_row.to_dict())

        elif label == "Y":
            if current_row is not None:
                # Y 이미지 이동
                dest_folder = os.path.join(dest_dir, f"{part_name}_Y불량검수")
                os.makedirs(dest_folder, exist_ok=True)
                dest_path = os.path.join(dest_folder, fname)
                shutil.move(img_path, dest_path)
                print(f"Moved (Y): {img_path} -> {dest_path}", flush=True)
                
                # Y 불량검수 처리된 이미지 정보 저장 (Y 이미지)
                processed_Y.append(current_row.to_dict())
                
                # 대응되는 이미지들도 처리
                corresponding_bad, corresponding_Y = process_corresponding_images(df, current_row, "Y", base_dir, dest_dir)
                processed_bad.extend(corresponding_bad)
                processed_Y.extend(corresponding_Y)
        
        elif label == "bad":
            if current_row is not None:
                # bad 이미지 이동
                dest_folder = os.path.join(dest_dir, f"{part_name}_불량검수")
                os.makedirs(dest_folder, exist_ok=True)
                dest_path = os.path.join(dest_folder, fname)
                shutil.move(img_path, dest_path)
                print(f"Moved (bad): {img_path} -> {dest_path}", flush=True)
                
                # bad 불량검수 처리된 이미지 정보 저장 (bad 이미지)
                processed_bad.append(current_row.to_dict())
                
                # 대응되는 이미지들도 처리
                corresponding_bad, corresponding_Y = process_corresponding_images(df, current_row, "bad", base_dir, dest_dir)
                processed_bad.extend(corresponding_bad)
                processed_Y.extend(corresponding_Y)
        else:
            print(f"[WARN] Unknown label: {label} for {img_path}", flush=True)
            cv2.destroyAllWindows()
            continue  # 혹시 모를 예외
        cv2.destroyAllWindows()
    elif key in [ord('f'), 57]:
        # 엑셀에서 병합파트명 가져오기
        fname = os.path.basename(img_path)
        row = df[df['이미지파일명'] == fname]
        if not row.empty:
            part_name = row.iloc[0]['병합파트명']
            current_row = row.iloc[0]
        else:
            part_name = "unknown"
            current_row = None
            
        sealing_folder = os.path.join(dest_dir, f"{part_name}_사용불가")
        os.makedirs(sealing_folder, exist_ok=True)
        dest_path = os.path.join(sealing_folder, os.path.basename(img_path))
        shutil.move(img_path, dest_path)
        print(f"Moved to sealing: {img_path} -> {dest_path}", flush=True)
        
        # 사용불가 처리된 이미지 정보 저장
        if current_row is not None:
            processed_unusable.append(current_row.to_dict())
            
            # bad 또는 Y 라벨인 경우 대응되는 이미지들도 사용불가로 처리
            if label in ["bad", "Y"]:
                corresponding_bad, corresponding_Y = process_corresponding_images(df, current_row, label, base_dir, dest_dir, is_unusable=True)
                # 대응되는 이미지들도 사용불가 리스트에 추가
                processed_unusable.extend(corresponding_bad)
                processed_unusable.extend(corresponding_Y)
        
    elif key == 27:  # esc
        print("[INFO] Stopped by user (ESC key)", flush=True)
        cv2.destroyAllWindows()
        break

# 프로그램 종료 시 엑셀 파일 생성 및 원본 업데이트
def create_excel_files():
    # 1. 양품검수 엑셀 파일 생성
    if processed_good:
        good_df = pd.DataFrame(processed_good)
        good_excel_path = os.path.join(dest_dir, f"{base_dir}_양품검수.xlsx")
        good_df.to_excel(good_excel_path, index=False)
        print(f"양품검수 엑셀 파일 생성: {good_excel_path}")
    
    # 2. bad 불량검수 엑셀 파일 생성
    if processed_bad:
        bad_df = pd.DataFrame(processed_bad)
        bad_excel_path = os.path.join(dest_dir, f"{base_dir}_불량검수.xlsx")
        bad_df.to_excel(bad_excel_path, index=False)
        print(f"bad 불량검수 엑셀 파일 생성: {bad_excel_path}")
    
    # 3. Y 불량검수 엑셀 파일 생성
    if processed_Y:
        Y_df = pd.DataFrame(processed_Y)
        Y_excel_path = os.path.join(dest_dir, f"{base_dir}_Y불량검수.xlsx")
        Y_df.to_excel(Y_excel_path, index=False)
        print(f"Y 불량검수 엑셀 파일 생성: {Y_excel_path}")
    
    # 4. 사용불가 엑셀 파일 생성
    if processed_unusable:
        unusable_df = pd.DataFrame(processed_unusable)
        unusable_excel_path = os.path.join(dest_dir, f"{base_dir}_사용불가.xlsx")  
        unusable_df.to_excel(unusable_excel_path, index=False)
        print(f"사용불가 엑셀 파일 생성: {unusable_excel_path}")
    
    # 5. 원본 엑셀 파일에서 처리된 이미지들 제거
    processed_filenames = []
    for item in processed_good + processed_bad + processed_Y + processed_unusable:
        processed_filenames.append(item['이미지파일명'])
    
    # 처리되지 않은 이미지들만 남기기
    remaining_df = df[~df['이미지파일명'].isin(processed_filenames)]
    
    # 원본 엑셀 파일 백업 후 업데이트
    backup_path = os.path.join(base_dir, f"후드_백업.xlsx")
    df.to_excel(backup_path, index=False)
    print(f"원본 엑셀 파일 백업: {backup_path}")
    
    remaining_df.to_excel(excel_path, index=False)
    print(f"원본 엑셀 파일 업데이트: {excel_path}")
    print(f"처리된 이미지 수: {len(processed_filenames)}")
    print(f"남은 이미지 수: {len(remaining_df)}")

# 프로그램 종료 시 엑셀 파일 생성
create_excel_files()