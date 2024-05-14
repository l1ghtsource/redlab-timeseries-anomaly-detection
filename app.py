import streamlit as st
import pandas as pd
import altair as alt

from ml.autoencoder import AER, AER_plot
from ml.clusterization_and_knn import K_Means, KNN_Anom, KMeans_plot, KNN_plot
from ml.adtk_algos import ADTK_KMeans, ADTK_OutlierDetector, ADTK_Pca, ADTK_plot
from ml.isolation_forest import IF, IF_plot

from ml.utils import getMean, getRange, getStd

methods_list = ['Isolation Forest', 'K-Means', 'Outlier Detector',
                'PCA', 'Autoencoder', 'K-Means (By Day)', 'KNN (By Day)']


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
    anomalies = AER(timeseries)

    fig = AER_plot(timeseries, anomalies)
    st.plotly_chart(fig)

    return pd.DataFrame(anomalies)


@st.cache_data
def find_anomalies_kmeans(timeseries):
    dd = K_Means(timeseries)
    fig = KMeans_plot(dd)
    st.plotly_chart(fig)

    return dd.loc[dd.anomaly_cls == 1, ['datetime', 'value']].reset_index().drop(columns='index').rename(
        columns={'datetime': 'timestamp'})


@st.cache_data
def find_anomalies_knn(timeseries):
    dd_ = KNN_Anom(timeseries)
    fig = KNN_plot(dd_)
    st.plotly_chart(fig)

    return dd_.loc[dd_['anomaly_knn'], ['datetime', 'value']].reset_index().drop(columns='index').rename(
        columns={'datetime': 'timestamp'})


@st.cache_data
def find_anomalies_iforest(timeseries):
    dd = IF(timeseries)
    df = dd[dd['anomaly'] == -1].drop(columns='anomaly').reset_index().drop(columns='index')
    anomalies = dd.drop(columns='value').set_index('timestamp')
    anomalies = anomalies['anomaly'].replace({-1: True, 1: False})
    ts = dd.drop(columns='anomaly').set_index('timestamp')

    fig = IF_plot(ts, anomalies)
    st.plotly_chart(fig)

    return df


@st.cache_data
def find_anomalies_adtk_kmeans(timeseries_idx):
    anomalies = ADTK_KMeans(timeseries_idx)
    anomaly_df = pd.DataFrame(anomalies)
    anomaly_df = anomaly_df.loc[anomaly_df[0] == 1]

    fig = ADTK_plot(timeseries_idx, anomalies, 'K-MEANS')
    st.plotly_chart(fig)

    idx = anomaly_df.index.tolist()
    return timeseries_idx.loc[idx].reset_index()


@st.cache_data
def find_anomalies_adtk_outlier_detector(timeseries_idx):
    anomalies = ADTK_OutlierDetector(timeseries_idx)
    anomaly_df = pd.DataFrame(anomalies)
    anomaly_df = anomaly_df.loc[anomaly_df[0] == 1]

    fig = ADTK_plot(timeseries_idx, anomalies, 'Outlier Detector')
    st.plotly_chart(fig)

    idx = anomaly_df.index.tolist()
    return timeseries_idx.loc[idx].reset_index()


@st.cache_data
def find_anomalies_adtk_pca(timeseries_idx):

    anomalies = ADTK_Pca(timeseries_idx)
    anomaly_df = pd.DataFrame(anomalies)
    anomaly_df = anomaly_df.loc[anomaly_df[0] == 1]

    fig = ADTK_plot(timeseries_idx, anomalies, 'PCA')
    st.plotly_chart(fig)

    idx = anomaly_df.index.tolist()
    return timeseries_idx.loc[idx].reset_index()


@st.cache_data
def get_info(timeseries, anomaly_df):
    range_of_series = getRange(timeseries)
    mean_val = getMean(timeseries)
    std_val = getStd(timeseries)
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
        timeseries = st.session_state["data"]

        with st.sidebar:
            st.title('📉 Поиск Аномалий во Временных Рядах')

            numeric_columns = timeseries.select_dtypes(include=['float', 'int'])
            numeric_column_names = numeric_columns.columns.tolist()

            selected_value = st.selectbox('Выберите столбец для поиска аномалий', numeric_column_names)
            selected_method = st.selectbox('Выберите режим поиска аномалий', methods_list)

            timeseries = pd.concat([timeseries['timestamp'], timeseries[selected_value]], axis=1)

            if st.button("Загрузить другой файл"):
                st.session_state["state"] = "initial"
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


if __name__ == "__main__":
    main()
