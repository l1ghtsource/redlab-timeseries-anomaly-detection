import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

from ml.autoencoder import AnomaliesAER
from ml.clusterization_and_knn import AnomaliesDetector
from ml.adtk_algos import AnomaliesADTK
from ml.isolation_forest import IsolationForestDetector
from ml.utils import TimeSeriesStatsCalculator


def download_csv(df):
    csv = df.to_csv(index=False)
    return csv


st.set_page_config(
    page_title='Поиск Аномалий во Временных Рядах',
    page_icon='📉',
    layout='wide',
    initial_sidebar_state='expanded'
)

alt.themes.enable('dark')


@st.cache_data
def find_anomalies_AER(timeseries):
    aer = AnomaliesAER(timeseries)
    anomalies = aer.detect_anomalies()

    fig = aer.plot_anomalies()
    st.plotly_chart(fig)

    return pd.DataFrame(anomalies)


@st.cache_data
def find_anomalies_kmeans(timeseries):
    detector = AnomaliesDetector(timeseries)
    kmeans_anomalies = detector.k_means()

    fig = detector.k_means_plot()
    st.plotly_chart(fig)

    anomalies = kmeans_anomalies.loc[kmeans_anomalies.anomaly_cls == 1, ['datetime', 'value']]
    return anomalies.reset_index().drop(columns='index').rename(columns={'datetime': 'timestamp'})


@st.cache_data
def find_anomalies_knn(timeseries):
    detector = AnomaliesDetector(timeseries)
    knn_anomalies = detector.knn_anom()

    fig = detector.knn_plot()
    st.plotly_chart(fig)

    anomalies = knn_anomalies.loc[knn_anomalies['anomaly_knn'], ['datetime', 'value']]
    return anomalies.reset_index().drop(columns='index').rename(columns={'datetime': 'timestamp'})


@st.cache_data
def find_anomalies_iforest(timeseries):
    detector = IsolationForestDetector()
    detector.fit(timeseries)

    fig = detector.plot()
    st.plotly_chart(fig)

    anomalies_df = detector.df[detector.df['anomaly'] == -1].drop(columns='anomaly').reset_index().drop(columns='index')

    return anomalies_df


@st.cache_data
def find_anomalies_adtk_kmeans(timeseries_idx):
    adtk = AnomaliesADTK(timeseries_idx)
    anomalies = adtk.detect_kmeans()
    anomaly_df = pd.DataFrame(anomalies)
    anomaly_df = anomaly_df.loc[anomaly_df[0] == 1]

    fig = adtk.plot_anomalies(anomalies, 'K-MEANS')
    st.plotly_chart(fig)

    idx = anomaly_df.index.tolist()
    return timeseries_idx.loc[idx].reset_index()


@st.cache_data
def find_anomalies_adtk_outlier_detector(timeseries_idx):
    adtk = AnomaliesADTK(timeseries_idx)
    anomalies = adtk.detect_outliers()
    anomaly_df = pd.DataFrame(anomalies)
    anomaly_df = anomaly_df.loc[anomaly_df[0] == 1]

    fig = adtk.plot_anomalies(anomalies, 'Outlier Detector')
    st.plotly_chart(fig)

    idx = anomaly_df.index.tolist()
    return timeseries_idx.loc[idx].reset_index()


@st.cache_data
def find_anomalies_adtk_pca(timeseries_idx):
    adtk = AnomaliesADTK(timeseries_idx)
    anomalies = adtk.detect_pca()
    anomaly_df = pd.DataFrame(anomalies)
    anomaly_df = anomaly_df.loc[anomaly_df[0] == 1]

    fig = adtk.plot_anomalies(anomalies, 'PCA')
    st.plotly_chart(fig)

    idx = anomaly_df.index.tolist()
    return timeseries_idx.loc[idx].reset_index()


@st.cache_data
def get_info(timeseries, anomaly_df):
    stats_calculator = TimeSeriesStatsCalculator(timeseries)

    range_of_series = stats_calculator.get_range()
    mean_val = stats_calculator.get_mean()
    std_val = stats_calculator.get_std()

    anomalies_count = len(anomaly_df)

    return anomalies_count, range_of_series, mean_val, std_val


def find_anomalies(method, timeseries):
    if method == 'Autoencoder':
        return find_anomalies_AER(timeseries)

    elif method == 'K-Means (By Day)':
        return find_anomalies_kmeans(timeseries)

    elif method == 'KNN (By Day)':
        return find_anomalies_knn(timeseries)

    elif method == 'Isolation Forest':
        return find_anomalies_iforest(timeseries)

    elif method == 'K-Means':
        timeseries_idx = timeseries.copy()
        timeseries_idx['timestamp'] = pd.to_datetime(timeseries_idx['timestamp'])
        timeseries_idx.set_index('timestamp', inplace=True)
        return find_anomalies_adtk_kmeans(timeseries_idx)

    elif method == 'Outlier Detector':
        timeseries_idx = timeseries.copy()
        timeseries_idx['timestamp'] = pd.to_datetime(timeseries_idx['timestamp'])
        timeseries_idx.set_index('timestamp', inplace=True)
        return find_anomalies_adtk_outlier_detector(timeseries_idx)

    elif method == 'PCA':
        timeseries_idx = timeseries.copy()
        timeseries_idx['timestamp'] = pd.to_datetime(timeseries_idx['timestamp'])
        timeseries_idx.set_index('timestamp', inplace=True)
        return find_anomalies_adtk_pca(timeseries_idx)


def main():
    state = st.session_state.get("state", "initial")

    if state == "initial":
        st.title('📉 Поиск Аномалий во Временных Рядах')

        uploaded_file = st.file_uploader("Выберите CSV файл", type=['csv'])
        if uploaded_file is not None:
            st.session_state["data"] = pd.read_csv(uploaded_file)
            st.session_state["state"] = "working"
            st.rerun()

    elif state == "working":
        timeseries_all = st.session_state["data"]

        with st.sidebar:
            st.title('📉 Поиск Аномалий во Временных Рядах')

            methods_list = [
                'Isolation Forest',
                # 'K-Means',
                # 'Outlier Detector',
                # 'PCA',
                'Autoencoder',
                # 'K-Means (By Day)',
                # 'KNN (By Day)'
            ]

            numeric_columns = timeseries_all.select_dtypes(include=['float', 'int'])
            numeric_column_names = numeric_columns.columns.tolist()

            selected_value = st.selectbox('Выберите столбец для поиска аномалий', numeric_column_names)
            selected_method = st.selectbox('Выберите режим поиска аномалий', methods_list)

            timeseries = pd.concat([timeseries_all['timestamp'], timeseries_all[selected_value]], axis=1)
            timeseries.rename(columns={selected_value: 'value'}, inplace=True)

            if st.button("Загрузить другой файл"):
                st.session_state["state"] = "initial"
                st.rerun()

            elif st.button("Анализ временного ряда"):
                st.session_state["state"] = "analyse"
                st.rerun()

        col = st.columns((1.5, 4.5, 2), gap='medium')

        anomaly_df = pd.DataFrame()

        with col[1]:
            st.markdown('#### Обнаруженные аномалии')
            anomaly_df = find_anomalies(selected_method, timeseries)

        with col[2]:
            st.markdown('#### Аномальные моменты')
            st.dataframe(anomaly_df, width=500)

            with st.expander('Подробнее о режимах', expanded=True):
                st.write('''
                    - Текст.
                    - :orange[**Еще текст**]: про что-то
                    - :orange[**И еще текст**]: бредовый
                    ''')

        with col[0]:
            st.markdown('#### Информация')

            anomalies_count, range_of_series, mean_val, std_val = get_info(timeseries, anomaly_df)

            st.metric(label=':white[Количество аномалий]',
                      value=anomalies_count)

            st.metric(label=':white[Размах]',
                      value=round(range_of_series, 4))

            st.metric(label=':white[Среднее]',
                      value=round(mean_val, 4))

            st.metric(label=':white[Отклонение]',
                      value=round(std_val, 4))

            st.markdown('#### Скачать отчёт')

            st.download_button(
                label="Скачать CSV",
                data=download_csv(anomaly_df),
                file_name=f'anomalies_{selected_method}.csv',
                mime='text/csv'
            )

    elif state == "analyse":
        timeseries_all = st.session_state["data"]

        with st.sidebar:
            st.title('📉 Анализ временного ряда')

            methods_list = [
                'Тепловая карта корелляционной матрицы',
                'Анализ причинно-следственных связей'
            ]

            selected_method = st.selectbox('Выберите режим работы', methods_list)

            if st.button("Загрузить другой файл"):
                st.session_state["state"] = "initial"
                st.rerun()

            elif st.button("Вернуться к поиску аномалий"):
                st.session_state["state"] = "working"
                st.rerun()

        if selected_method == 'Тепловая карта корелляционной матрицы':
            stats_calculator = TimeSeriesStatsCalculator(timeseries_all)
            corr_matrix = stats_calculator.get_correlation().round(2)

            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="blugrn")
            fig.update_layout(title='Тепловая карта корелляционной матрицы')

            st.plotly_chart(fig)

            corr_pairs = corr_matrix > 0.9

            pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_pairs.iloc[i, j]:
                        pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

            res = 'Пары с корелляцией > 0.9: '

            for pair in pairs:
                res += f'{pair[0]} и {pair[1]} ({pair[2]})'

            st.write(res)


if __name__ == "__main__":
    main()
