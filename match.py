"""
엑셀파일과 폴더의 파일명 매칭을 확인하는 프로그램
매칭만 확인
엑셀 및 파일명 변경 없음

start_row로 엑셀 내에서 읽어오기 시작할 행 설정

"""




import pandas as pd
import os
from pathlib import Path

def check_csv_data_matching():
    """
    CSV 파일과 10개 폴더의 파일명 매칭을 확인하는 함수
    """
    # 파일 경로 설정
    excel_path = "quality_6월16~7월15일.xlsx"
    folders = ['0715','0716','0717']
    
    try:
        # 엑셀 파일 읽기 (특정 행부터 끝까지)
        print(" 파일을 읽는 중...")
        # 시작 행 설정 (일련번호가 아니라 엑셀 기준 행 번호)
        #start_row = 1
        start_row = 2942
        df = pd.read_excel(excel_path, header=0, skiprows=range(1, start_row-1))
        print(f"엑셀 파일 로드 완료: {len(df)} 행 ({start_row}행부터 끝까지 읽음)")
        print(df.head(1))



        image_column = '이미지파일명'
        
        # CSV에서 파일명 추출 (확장자 제거)
        csv_files = set()
        for filename in df[image_column].dropna():
            # 확장자 제거
            name_without_ext = os.path.splitext(str(filename))[0]
            csv_files.add(name_without_ext)
        
        print(f"엑셀에서 추출한 파일명 수: {len(csv_files)}")
        
        # 날짜 폴더에서 파일명 추출
        all_folder_files = set()
        folder_files_dict = {}
        
        for folder in folders:
            if not os.path.exists(folder):
                print(f"⚠️  폴더 '{folder}'가 존재하지 않습니다.")
                continue
            
            folder_files = set()
            for filename in os.listdir(folder):
                if filename.lower().endswith(('.jpg')):
                    name_without_ext = os.path.splitext(filename)[0]
                    folder_files.add(name_without_ext)
                    all_folder_files.add(name_without_ext)
            
            folder_files_dict[folder] = folder_files
            print(f"📁 {folder} 폴더: {len(folder_files)}개 파일")
        
        print(f"전체 폴더에서 추출한 파일명 수: {len(all_folder_files)}")
        
        # 매칭 분석
        matched_files = csv_files & all_folder_files  # 교집합
        csv_only = csv_files - all_folder_files       # CSV에만 있는 파일
        folder_only = all_folder_files - csv_files    # 폴더에만 있는 파일
        
        # 결과 출력
        print("\n" + "="*60)
        print("📊 파일 매칭 분석 결과")
        print("="*60)
        
        print(f"\n✅ 매칭된 파일: {len(matched_files)}개")
        print(f"❌ CSV에만 있는 파일: {len(csv_only)}개")
        print(f"⚠️  폴더에만 있는 파일: {len(folder_only)}개")
        
        # CSV에만 있는 파일들 출력
        if csv_only:
            print(f"\n❌ CSV에만 있는 파일들 ({len(csv_only)}개):")
            for i, filename in enumerate(sorted(csv_only), 1):
                print(f"   {i:3d}. {filename}")
                if i >= 20:  # 처음 20개만 출력
                    print(f"   ... 외 {len(csv_only) - 20}개")
                    break
        else:
            print("\n✅ CSV에만 있는 파일 없음")
        
        # 폴더에만 있는 파일들 출력
        if folder_only:
            print(f"\n⚠️  폴더에만 있는 파일들 ({len(folder_only)}개):")
            for i, filename in enumerate(sorted(folder_only), 1):
                print(f"   {i:3d}. {filename}")
                if i >= 20:  # 처음 20개만 출력
                    print(f"   ... 외 {len(folder_only) - 20}개")
                    break
        else:
            print("\n✅ 폴더에만 있는 파일 없음")
        
        
        # 요약
        print(f"\n🎯 요약:")
        if len(csv_only) == 0 and len(folder_only) == 0:
            print("   🎉 완벽한 매칭! 모든 파일이 일치합니다.")
        elif len(csv_only) == 0:
            print("   ✅ CSV의 모든 파일이 폴더에 존재합니다.")
        elif len(folder_only) == 0:
            print("   ✅ 폴더의 모든 파일이 CSV에 기록되어 있습니다.")
        else:
            print("   ⚠️  일부 파일이 매칭되지 않았습니다.")
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {excel_path}")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    check_csv_data_matching()
