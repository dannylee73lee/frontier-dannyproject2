import os
import pandas as pd
import folium
from folium import DivIcon
import streamlit as st
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import matplotlib.font_manager as fm



# 폰트 설정 함수
def configure_font():
    font_path = 'NanumGothic.ttf'  # 저장한 NanumGothic.ttf 폰트 파일 경로
    fm.fontManager.addfont(font_path)  # matplotlib에 폰트 등록
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()  # 폰트 설정

# 파일 로드 함수
def load_data(file_path, additional_file_path):
    try:
        df = pd.read_csv(file_path)
        additional_df = pd.read_csv(additional_file_path)

        # 데이터 전처리
        df = df[['bld_address', 'cluster', 'bld_lat', 'bld_lon', 'dt', 'nr_dr', 'nr_ul', 'lte_dl', 'lte_ul', 'congestion', 'unique_imsi', 'nr_cell_nm', 'lte_cell_nm']]
        additional_df = additional_df[['dt', 'cluster_nm', 'aau_nm', 'max_peak_dl_prb_1h', 'target_prb_1h']]
        
        # NaN 값 처리
        df = df.dropna(subset=['bld_lat', 'bld_lon']).fillna('')
        additional_df = additional_df.dropna(subset=['max_peak_dl_prb_1h', 'target_prb_1h']).fillna('')
        
        # 날짜 형식 변환
        df['dt'] = pd.to_datetime(df['dt'], format='%Y%m%d')
        additional_df['dt'] = pd.to_datetime(additional_df['dt'], format='%Y%m%d')

        return df, additional_df
    except FileNotFoundError as e:
        st.error(f"파일을 찾을 수 없습니다: {e}")
        return None, None
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
        return None, None

# 색상 맵핑 함수
def get_color(nr_dr, nr_ul, lte_dl, lte_ul):
    traffic = nr_dr + nr_ul + lte_dl + lte_ul
    if traffic < 500:
        return 'green'
    elif traffic < 800:
        return 'orange'
    else:
        return 'red'

# 원 크기 맵핑 함수 (더 세밀하게 조절)
def get_radius(nr_dr, nr_ul, lte_dl, lte_ul):
    traffic = nr_dr + nr_ul + lte_dl + lte_ul
    return max(5, min(traffic / 30, 30))  # 최소 5, 최대 30


# 팝업 텍스트 생성 함수 (CSS 스타일 더 축소)
def create_popup_text(row):
    def safe_format(value):
        return f"{round(value, 2):,}" if pd.notnull(value) else 'N/A'

    # 폰트 크기 및 셀 간 간격을 최소화하여 테이블 높이를 줄임
    popup_text = (
        f"<table style='font-size: 11px; width: 100%; max-width: 250px; border: 1px solid black; border-collapse: collapse; margin: 0; padding: 0;'>"
        f"<tr style='background-color: #f2f2f2;'><th style='border: 1px solid black; padding: 1px;'>구분</th><th style='border: 1px solid black; padding: 1px;'>항목</th><th style='border: 1px solid black; padding: 1px;'>값</th></tr>"
        f"<tr><td style='border: 1px solid black; padding: 1px;' rowspan='2'>5G</td><td style='border: 1px solid black; padding: 1px;'>DL [GB]</td><td style='border: 1px solid black; padding: 1px;'>{safe_format(row['nr_dr'])}</td></tr>"
        f"<tr><td style='border: 1px solid black; padding: 1px;'>UL [GB]</td><td style='border: 1px solid black; padding: 1px;'>{safe_format(row['nr_ul'])}</td></tr>"
        f"<tr><td style='border: 1px solid black; padding: 1px;' rowspan='2'>LTE</td><td style='border: 1px solid black; padding: 1px;'>DL [GB]</td><td style='border: 1px solid black; padding: 1px;'>{safe_format(row['lte_dl'])}</td></tr>"
        f"<tr><td style='border: 1px solid black; padding: 1px;'>UL [GB]</td><td style='border: 1px solid black; padding: 1px;'>{safe_format(row['lte_ul'])}</td></tr>"
        f"<tr style='background-color: #f2f2f2;'><td style='border: 1px solid black; padding: 1px;' rowspan='2'>소계</td><td style='border: 1px solid black; padding: 1px;'>5G [GB]</td><td style='border: 1px solid black; padding: 1px;'>{safe_format(row['nr_dr'] + row['nr_ul'])}</td></tr>"
        f"<tr style='background-color: #f2f2f2;'><td style='border: 1px solid black; padding: 1px;'>LTE [GB]</td><td style='border: 1px solid black; padding: 1px;'>{safe_format(row['lte_dl'] + row['lte_ul'])}</td></tr>"
        f"</table>"
    )
    return popup_text


# 팝업을 표시할 때 IFrame 크기를 테이블 크기에 맞게 조정
def create_marker_with_popup(row):
    popup_text = create_popup_text(row)
    
    # IFrame의 높이를 고정하지 않고 팝업 최대 너비만 설정하여 자동으로 조정
    iframe = folium.IFrame(popup_text, width=200)  # 가로 200px 설정, height 생략
    popup = folium.Popup(iframe, max_width=200, max_height=100)  # 팝업 최대 크기만 설정하여 테이블 크기에 맞게 조정
    marker = folium.CircleMarker(
        location=[row['bld_lat'], row['bld_lon']],
        radius=get_radius(row['nr_dr'], row['nr_ul'], row['lte_dl'], row['lte_ul']),
        color=get_color(row['nr_dr'], row['nr_ul'], row['lte_dl'], row['lte_ul']),
        fill=True,
        fill_color=get_color(row['nr_dr'], row['nr_ul'], row['lte_dl'], row['lte_ul']),
        fill_opacity=0.7,
        popup=popup
    )
    return marker


# 달성도 테이블 생성 및 지도에 표시하는 함수
def add_achievement_table_to_map(m, df_additional, selected_dates, selected_clusters):
    achievement_data = []
    
    # 선택한 클러스터 및 날짜에 따른 데이터 필터링 및 달성도 계산
    for cluster_nm in selected_clusters:
        cluster_data = df_additional[(df_additional['cluster_nm'] == cluster_nm) & (df_additional['dt'].isin(pd.to_datetime(selected_dates)))]
        for aau_nm in cluster_data['aau_nm'].unique():
            subset = cluster_data[cluster_data['aau_nm'] == aau_nm]
            if not subset.empty:
                target_prb_1h = subset['target_prb_1h'].iloc[0]
                current_level = subset['max_peak_dl_prb_1h'].mean()
                achievement = (1 - (current_level - target_prb_1h) / target_prb_1h) * 100
                achievement_data.append({
                    "AAU 명": aau_nm,
                    "목표 PRB": target_prb_1h,
                    "현재 수준": current_level,
                    "달성도 (%)": round(achievement, 2)
                })

    # 테이블을 HTML로 변환 (테두리 색 회색)
    table_html = """
    <table style='font-size: 12px; width: auto; border: 1px solid grey; border-collapse: collapse;'>
    <thead style='background-color: #f2f2f2;'>
        <tr>
            <th style='border: 1px solid grey;'>AAU 명</th>
            <th style='border: 1px solid grey;'>목표 PRB</th>
            <th style='border: 1px solid grey;'>현재 수준</th>
            <th style='border: 1px solid grey;'>달성도 (%)</th>
        </tr>
    </thead>
    <tbody>
    """
    for entry in achievement_data:
        table_html += f"""
        <tr>
            <td style='border: 1px solid grey;'>{entry['AAU 명']}</td>
            <td style='border: 1px solid grey;'>{entry['목표 PRB']}</td>
            <td style='border: 1px solid grey;'>{entry['현재 수준']}</td>
            <td style='border: 1px solid grey;'>{entry['달성도 (%)']}</td>
        </tr>
        """
    table_html += "</tbody></table>"

    # 지도에 HTML 테이블 추가 (위치: 왼쪽 아래)
    folium.Marker(
        location=[m.location[0] + 0.005, m.location[1] - 0.002],  # 위치 조정 (왼쪽 아래)
        icon=DivIcon(
            icon_size=(250, 250),  # 크기 조정 (자동 맞춤)
            icon_anchor=(0, 0),  # 왼쪽 위에 맞추기
            html=f'<div style="background-color: white; border: 1px solid grey; padding: 10px;">{table_html}</div>'
        )
    ).add_to(m)




# 지도 생성 함수에서 호출하는 부분은 동일
def create_map(df, selected_dates, selected_clusters, df_additional):
    selected_dates = pd.to_datetime(selected_dates)
    filtered_df = df[df['dt'].isin(selected_dates) & df['cluster'].isin(selected_clusters)].copy()

    filtered_df['total_traffic'] = filtered_df['nr_dr'] + filtered_df['nr_ul'] + filtered_df['lte_dl'] + filtered_df['lte_ul']
    top_5_df = filtered_df.nlargest(5, 'total_traffic')

    col1, col2 = st.columns([3, 1])

    with col1:
        if not filtered_df.empty:
            center_lat = filtered_df['bld_lat'].mean()
            center_lon = filtered_df['bld_lon'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=16, width='100%', height='600px')

            for idx, row in filtered_df.iterrows():
                color = get_color(row['nr_dr'], row['nr_ul'], row['lte_dl'], row['lte_ul'])
                radius = get_radius(row['nr_dr'], row['nr_ul'], row['lte_dl'], row['lte_ul'])
                popup_text = create_popup_text(row)
                iframe = folium.IFrame(popup_text, width=400, height=300)
                popup = folium.Popup(iframe, max_width=300)
                marker = folium.CircleMarker(
                    location=[row['bld_lat'], row['bld_lon']],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=popup
                )
                marker.add_to(m)

            for rank, (idx, row) in enumerate(top_5_df.iterrows(), start=1):
                folium.Marker(
                    location=[row['bld_lat'], row['bld_lon']],
                    icon=DivIcon(
                        icon_size=(30, 30),
                        icon_anchor=(15, 15),
                        html=f'<div style="font-size: 12px; color: white; text-align: center; '
                             f'background-color: transparent; border-radius: 50%; width: 30px; height: 30px; line-height: 30px;">{rank}</div>'
                    )
                ).add_to(m)

            # 달성도 테이블 지도에 추가
            add_achievement_table_to_map(m, df_additional, selected_dates, selected_clusters)

            folium_static(m)
        else:
            st.warning("선택한 날짜에 해당하는 데이터가 없습니다.")

    with col2:
        st.markdown("""
            <div style="background-color:white; border:1px solid black; padding:10px; width:160px; font-size:12px;">
            <b>Traffic 범위</b><br>
            <i style="background:green; width: 12px; height: 12px; display: inline-block;"></i> 0 - 500미만 [GB]<br>
            <i style="background:orange; width: 12px; height: 12px; display: inline-block;"></i> 500이상 - 800미만 [GB]<br>
            <i style="background:red; width: 12px; height: 12px; display: inline-block;"></i> 800+ [GB]<br>
            </div>
        """, unsafe_allow_html=True)


# 그래프 생성 함수
def create_graph(additional_df, selected_dates, cluster_nm):
    # 필터링된 데이터가 있는지 확인
    cluster_data = additional_df[additional_df['cluster_nm'] == cluster_nm]
    
    if cluster_data.empty:
        st.warning(f"클러스터 {cluster_nm}에 대한 데이터가 없습니다.")
        return None
    
    plt.figure(figsize=(12, 8))

    # 선택한 날짜를 datetime 형식으로 변환
    selected_dates = pd.to_datetime(selected_dates)
    
    # 각 aau_nm 별로 그래프를 그림
    for aau_nm in cluster_data['aau_nm'].unique():
        subset = cluster_data[cluster_data['aau_nm'] == aau_nm]
        if subset.empty:
            continue  # 데이터가 없는 경우 건너뛰기
        # 동그라미 모양으로 표시 추가
        plt.plot(subset['dt'], subset['max_peak_dl_prb_1h'], label=aau_nm, linewidth=3, marker='o')  # marker='o'는 동그라미

        # 각 데이터 포인트 중에서 선택한 날짜에 해당하는 데이터만 값 표시
        for x, y in zip(subset['dt'], subset['max_peak_dl_prb_1h']):
            if x in selected_dates:
                plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')  # 폰트 크기 14, 굵게 표시

    # 선택한 날짜 범위를 강조
    if selected_dates is not None:
        for date in selected_dates:
            if not isinstance(date, pd.Timestamp):
                date = pd.to_datetime(date)
            start_date = date
            end_date = start_date + pd.Timedelta(days=1)
            plt.axvspan(start_date, end_date, color='yellow', alpha=0.3)

    plt.xlabel('Date', fontsize=13)
    plt.ylabel('Max Peak DL PRB 1H', fontsize=13)
    plt.title(f'Max Peak DL PRB 1H over Time for {cluster_nm}', fontsize=18, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True)

    # x축 날짜 포맷 및 간격 설정
    ax = plt.gca()

    # 날짜 간격을 하루 단위로 설정 (필요시 WeekdayLocator로 변경 가능)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 1일 단위로 날짜 간격 설정
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))  # 날짜 포맷 설정

    plt.xticks(rotation=45, ha='right', fontsize=14)  # x축 라벨 회전 및 크기 조정
    plt.yticks(fontsize=13)
    plt.tight_layout()

    # 그래프를 이미지로 저장
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    return img_str


# 상위 Top 5 데이터를 표시하는 함수
def display_top5_data(df, selected_dates, selected_clusters):
    # 선택한 날짜와 클러스터로 데이터 필터링
    filtered_df = df[df['dt'].isin(pd.to_datetime(selected_dates)) & df['cluster'].isin(selected_clusters)].copy()

    # 5G 데이터 피벗 테이블 생성
    pivot_5g = pd.pivot_table(filtered_df, index='nr_cell_nm', values=['nr_dr', 'nr_ul'], aggfunc='sum')

    # LTE 데이터 피벗 테이블 생성
    pivot_lte = pd.pivot_table(filtered_df, index='lte_cell_nm', values=['lte_dl', 'lte_ul'], aggfunc='sum')

    # 5G에서 상위 5개 데이터 추출 (다운로드와 업로드의 합계 기준)
    pivot_5g['total_traffic'] = pivot_5g['nr_dr'] + pivot_5g['nr_ul']
    top_5g = pivot_5g.nlargest(5, 'total_traffic').drop(columns='total_traffic')

    # 소수점 첫째 자리까지만 표시
    top_5g = top_5g.round(1)

    # 5G 테이블 열 이름 변경
    top_5g = top_5g.rename(columns={'nr_dr': '5G DL[GB]', 'nr_ul': '5G UL[GB]'})
    top_5g.index.name = '셀 이름'  # 인덱스명을 '셀 이름'으로 변경

    # LTE에서 상위 5개 데이터 추출 (다운로드와 업로드의 합계 기준)
    pivot_lte['total_traffic'] = pivot_lte['lte_dl'] + pivot_lte['lte_ul']
    top_lte = pivot_lte.nlargest(5, 'total_traffic').drop(columns='total_traffic')

    # 소수점 첫째 자리까지만 표시
    top_lte = top_lte.round(1)

    # LTE 테이블 열 이름 변경
    top_lte = top_lte.rename(columns={'lte_dl': 'LTE DL[GB]', 'lte_ul': 'LTE UL[GB]'})
    top_lte.index.name = '셀 이름'  # 인덱스명을 '셀 이름'으로 변경

    # 두 개의 열을 나란히 표시
    col1, col2 = st.columns(2)

    with col1:
        # subheader 글자 크기를 줄여서 표시
        st.markdown("<h4 style='font-size:16px;'>Top 5 5G 데이터</h4>", unsafe_allow_html=True)
        st.dataframe(top_5g)

    with col2:
        # subheader 글자 크기를 줄여서 표시
        st.markdown("<h4 style='font-size:16px;'>Top 5 LTE 데이터</h4>", unsafe_allow_html=True)
        st.dataframe(top_lte)

# 수정된 run 함수
def run():
    # 폰트 설정
    configure_font()

    # 파일 경로
    file_path = 'sample_dna 0901_0930.csv'
    additional_file_path = 'sample_daily 0901_0930.csv'

    # 데이터 로드
    df, additional_df = load_data(file_path, additional_file_path)
    if df is None or additional_df is None:
        return

    # 날짜와 클러스터 선택
    available_dates = sorted(df['dt'].dt.strftime('%Y-%m-%d').unique())
    selected_dates = st.multiselect('날짜 선택:', available_dates)
    selected_clusters = st.multiselect('클러스터 선택:', sorted(df['cluster'].unique()))

    if selected_dates and selected_clusters:
        # 상위 5개 데이터를 표시
        display_top5_data(df, selected_dates, selected_clusters)

        # 지도 생성 및 그래프 그리기
        create_map(df, selected_dates, selected_clusters, additional_df)

        for cluster in selected_clusters:
            img_str = create_graph(additional_df, selected_dates, cluster)
            st.image(f"data:image/png;base64,{img_str}", use_column_width=True)

# main 함수 호출 부분은 동일
if __name__ == "__main__":
    run()


