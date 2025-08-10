import pandas as pd
# import argparse # 파일 실행 시 

def setup_dataset(year: int, building_name: str):
    """
    Function: setup_dataset
        1. 토론토 에너지 데이터를 불러와 날짜 열 형식 변환
        2. 연도별로 필터링
        3. 토론토 날씨 데이터를 불러와 날짜 열 형식 변환
        4. 에너지 데이터와 날씨 데이터를 날짜 기준으로 병합
        5. 결과를 CSV로 저장
    Parameters:
        year (int): 처리할 연도
        building_name (str): 건물 이름 (예: Panther_parking_Lorriane)
    Return values:
        total_data (pd.DataFrame): 병합된 데이터프레임
    """
    # 토론토 에너지 데이터 불러오기
    elec_path = "C:/Users/ryudo/OneDrive - gachon.ac.kr/AiCE2/석사논문/Transformer/BDG2/data"
    elec = pd.read_csv(elec_path + "/electricity.csv", parse_dates=["timestamp"]) # parse_dates: timestamp 열을 datetime 형식으로 변환

    # timestamp → Date/Time(날짜만), Hour 분리
    elec["Date/Time"] = elec["timestamp"].dt.date
    elec["year"] = elec["timestamp"].dt.year
    elec["Hour"] = elec["timestamp"].dt.time

    # 연도 필터링
    elec = elec[elec["year"] == year].reset_index(drop=True)

    # 필요한 열만 선택 후 이름 변경
    elec = elec[["timestamp", "Date/Time", f"{building_name}"]]
    elec.rename(columns={f"{building_name}": "electricity"}, inplace=True)

    # 날씨 데이터 불러오기
    weather_path = f"C:/Users/ryudo/OneDrive - gachon.ac.kr/AiCE2/석사논문/Transformer/BDG2/data/toronto_weather/toronto_weather_{year}.csv"
    weather = pd.read_csv(weather_path, parse_dates=["Date/Time"])
    weather = weather[weather["Year"] == year].reset_index(drop=True)
    weather = weather[["Date/Time", "Mean Temp (°C)", "Total Rain (mm)"]]

    # 날짜 열 형식 변환
    weather["Date/Time"] = weather["Date/Time"].dt.date

    # 날짜 기준으로 병합
    total_data = pd.merge(elec, weather, on="Date/Time", how="inner")

    # 최종 열 선택
    total_data = total_data[["timestamp", "Date/Time", "electricity", "Mean Temp (°C)", "Total Rain (mm)"]]

    # 시간 순 정렬
    total_data.sort_values("timestamp", inplace=True)
    total_data.reset_index(drop=True, inplace=True)

    # 결과 저장
    total_data.to_csv(elec_path + f"/toronto_data_{year}.csv", index=False)
    print(f"[저장 완료 | Building: {building_name}, Year: {year}]: {elec_path}/toronto_data_{year}.csv")
    return total_data

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--year", type=int, required=True, help="처리할 연도")
#     parser.add_argument("--building_name", type=str, required=True, help="건물 이름 (예: Panther_parking_Lorriane)")
#     args = parser.parse_args()

#     toronto_total_data_preparation(args.year, args.building_name)

# # run example: python data_concat.py --year 2017 --building_name Panther_parking_Lorriane