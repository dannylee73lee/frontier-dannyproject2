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
import time

# 폰트 설정 함수
def configure_font():
    font_path = './NanumGothic.ttf'
    
    # 파일 존재 여부 확인
    if not os.path.exists(font_path):
        st.error(f"폰트 파일을 찾을 수 없습니다: {font_path}")
        return
    
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


# 팝업 텍스트 생성 함수
def create_popup_text(row):
    def safe_format(value):
        return f"{round(value, 2):,}" if pd.notnull(value) else 'N/A'

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


# 지도에 마커 추가하는 함수
def create_marker_with_popup(row):
    popup_text = create_popup_text(row)
    
    iframe = folium.IFrame(popup_text, width=200)  # 가로 200px 설정, height 생략
    popup = folium.Popup(iframe, max_width=200, max_height=100)
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


# 달성도 테이블 생성 및 지도에 추가하는 함수
def add_achievement_table_to_map(m, df_additional, selected_dates, selected_clusters):
    achievement_data = []
    
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

    folium.Marker(
        location=[m.location[0] + 0.005, m.location[1] - 0.002],
        icon=DivIcon(
            icon_size=(250, 250),
            icon_anchor=(0, 0),
            html=f'<div style="background-color: white; border: 1px solid grey; padding: 10px;">{table_html}</div>'
        )
    ).add_to(m)


# 프로그레스바 업데이트 함수 (데이터 처리와 연동)
def update_progress_bar(total_steps):
    progress_bar = st.progress(0)
    for step in range(total_steps):
        time.sleep(1)  # 지연 시간을 설정 (데이터 처리 시간이 짧을 경우 조정)
        progress_bar.progress((step + 1) / total_steps)
    progress_bar.empty()

# 사용자 입력을 받아오는 함수
def gpt_style_input(df):
    available_dates = sorted(df['dt'].dt.strftime('%Y-%m-%d').unique())
    # st.write("날짜를 선택해주세요.")
    selected_dates = st.multiselect('날짜 선택:', available_dates)

    available_clusters = sorted(df['cluster'].unique())
    # st.write("분석할 클러스터를 선택해주세요.")
    selected_clusters = st.multiselect('클러스터 선택:', available_clusters)

    # 입력받은 날짜와 클러스터 수에 맞춰 프로그레스바 업데이트
    total_steps = len(selected_dates) * len(available_clusters)
    update_progress_bar(total_steps)

    return selected_dates, selected_clusters


# 지도 생성 함수
# 지도 생성 함수
def create_map_with_progress_bar(df, selected_dates, selected_clusters, df_additional):
    selected_dates = pd.to_datetime(selected_dates)
    filtered_df = df[df['dt'].isin(selected_dates) & df['cluster'].isin(selected_clusters)].copy()

    filtered_df['total_traffic'] = filtered_df['nr_dr'] + filtered_df['nr_ul'] + filtered_df['lte_dl'] + filtered_df['lte_ul']
    top_5_df = filtered_df.nlargest(5, 'total_traffic')

    # 프로세스바 생성
    progress_bar = st.progress(0)
    total_steps = len(filtered_df) + len(top_5_df)  # 총 스텝 수는 마커 생성의 수와 상위 5개의 수로 설정

    col1, col2 = st.columns([3, 1])

    with col1:
        if not filtered_df.empty:
            center_lat = filtered_df['bld_lat'].mean()
            center_lon = filtered_df['bld_lon'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=16, width='100%', height='600px')

            # 데이터가 많을 경우 프로그레스바 업데이트
            for idx, row in enumerate(filtered_df.iterrows(), start=1):
                marker = create_marker_with_popup(row[1])
                marker.add_to(m)

                # 프로그레스바 업데이트
                progress_bar.progress(idx / total_steps)
                time.sleep(0.01)  # 느리게 작동하도록 지연 시간 설정

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

                # 프로그레스바 업데이트
                progress_bar.progress((len(filtered_df) + rank) / total_steps)
                time.sleep(0.01)  # 지연 시간 설정

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

    # 프로그레스바 완료 후 삭제
    progress_bar.empty()


# 그래프 생성 함수
def create_graph_gpt_style(additional_df, selected_dates, cluster_nm):
    cluster_data = additional_df[additional_df['cluster_nm'] == cluster_nm]
    
    if cluster_data.empty:
        st.warning(f"클러스터 {cluster_nm}에 대한 데이터가 없습니다.")
        return None
    
    plt.figure(figsize=(12, 8))

    selected_dates = pd.to_datetime(selected_dates)
    
    for aau_nm in cluster_data['aau_nm'].unique():
        subset = cluster_data[cluster_data['aau_nm'] == aau_nm]
        if subset.empty:
            continue
        plt.plot(subset['dt'], subset['max_peak_dl_prb_1h'], label=aau_nm, linewidth=3, marker='o')

        for x, y in zip(subset['dt'], subset['max_peak_dl_prb_1h']):
            if x in selected_dates:
                plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

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

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=13)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    return img_str


# 상위 5개 데이터를 표시하는 함수
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


# 최종 실행 함수
# 최종 실행 함수에서 수정된 지도 생성 함수 호출
def run_gpt_style():
    # 폰트 설정
    configure_font()

    # 메인 제목 표시
    st.markdown("<h3 style='text-align: center; font-size:24px;'>AI기반 밀집Cluster 품질 센싱 및 트래픽 자동분석</h3>", unsafe_allow_html=True)

    # 파일 경로 설정
    file_path = './sample_dna 0901_0930.csv'
    additional_file_path = './sample_daily 0901_0930.csv'

    # 데이터 로드
    df, additional_df = load_data(file_path, additional_file_path)
    if df is None or additional_df is None:
        return

    # 사용자 입력 유도 (df를 인자로 전달)
    selected_dates, selected_clusters = gpt_style_input(df)

    if selected_dates and selected_clusters:
        st.write("선택하신 데이터로 상위 5개 항목을 분석하여 표시합니다.")
        display_top5_data(df, selected_dates, selected_clusters)

        # 프로그레스바와 함께 지도를 생성
        st.write("선택한 날짜와 클러스터를 바탕으로 지도를 생성합니다.")
        create_map_with_progress_bar(df, selected_dates, selected_clusters, additional_df)

        st.write("선택된 클러스터의 트래픽 그래프를 보여드립니다.")
        for cluster in selected_clusters:
            img_str = create_graph_gpt_style(additional_df, selected_dates, cluster)
            st.image(f"data:image/png;base64,{img_str}", use_column_width=True)

# main 함수 호출
if __name__ == "__main__":
    run_gpt_style()
