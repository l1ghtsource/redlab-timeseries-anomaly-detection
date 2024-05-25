import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

from ml.autoencoder import AnomaliesAER
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
    anomalies = pd.DataFrame(aer.detect_anomalies())

    return anomalies


@st.cache_data
def find_anomalies_iforest(timeseries, start, end):
    detector = IsolationForestDetector()
    detector.fit(timeseries)

    df = detector.df[(pd.to_datetime(detector.df['timestamp']) >= start)
                     & (pd.to_datetime(detector.df['timestamp']) <= end)]

    anomalies_df = df[df['anomaly'] == -1].drop(columns='anomaly').reset_index().drop(columns='index')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df['value'], mode='lines', name='Value'))

    anomaly_dates = df[df['anomaly'] == -1].index
    fig.add_trace(go.Scatter(x=anomaly_dates, y=df.loc[anomaly_dates]['value'],
                             mode='markers', marker=dict(color='red'), name='Anomalies'))

    fig.update_layout(title=f'Anomalies in Time Series [Isolation Forest]',
                      xaxis_title='Datetime',
                      yaxis_title='Value')

    st.plotly_chart(fig)

    return anomalies_df


@st.cache_data
def get_info(timeseries, anomaly_df):
    stats_calculator = TimeSeriesStatsCalculator(timeseries)

    range_of_series = stats_calculator.get_range()
    mean_val = stats_calculator.get_mean()
    std_val = stats_calculator.get_std()

    anomalies_count = len(anomaly_df)

    return anomalies_count, range_of_series, mean_val, std_val


def find_anomalies(method, timeseries, start, end):
    if method == 'Autoencoder':
        anomalies = find_anomalies_AER(timeseries)

        timeseries['timestamp'] = pd.to_datetime(timeseries['timestamp'])
        anomalies['start'] = pd.to_datetime(anomalies['start'], unit='s')
        anomalies['end'] = pd.to_datetime(anomalies['end'], unit='s')

        timeseries = timeseries[(timeseries['timestamp'] >= start) & (timeseries['timestamp'] <= end)]
        anomalies = anomalies[(anomalies['start'] >= start) & (anomalies['end'] <= end)]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=timeseries['timestamp'],
                                 y=timeseries['value'],
                                 mode='lines',
                                 name='Time Series'))

        for _, row in anomalies.iterrows():
            anomaly_range = timeseries[(timeseries['timestamp'] >= row['start'])
                                       & (timeseries['timestamp'] <= row['end'])]
            fig.add_trace(go.Scatter(x=anomaly_range['timestamp'],
                                     y=anomaly_range['value'],
                                     mode='lines',
                                     line=dict(color='red')))

        fig.update_layout(xaxis_title='Time',
                          yaxis_title='Value',
                          title='Anomalies in Time Series [AER]')

        st.plotly_chart(fig)

        return anomalies

    elif method == 'Isolation Forest':
        return find_anomalies_iforest(timeseries, start, end)

    elif method == 'Multidimensional':
        pass


def main():
    state = st.session_state.get("state", "initial")

    if state == "initial":
        st.title('📉 Поиск Аномалий во Временных Рядах')

        uploaded_file = st.file_uploader("Выберите CSV файл", type=['csv'])
        if uploaded_file is not None:
            st.session_state["data"] = pd.read_csv(uploaded_file)
            st.session_state["state"] = "working"
            st.session_state["start"] = pd.to_datetime(st.session_state["data"]['timestamp'].min())
            st.session_state["end"] = pd.to_datetime(st.session_state["data"]['timestamp'].max())
            st.rerun()

        if st.button("Использовать логи из БД"):
            st.session_state["data"] = pd.read_csv(r'data\all_in_one.csv')
            st.session_state["state"] = "working"
            st.session_state["start"] = pd.to_datetime(st.session_state["data"]['timestamp'].min())
            st.session_state["end"] = pd.to_datetime(st.session_state["data"]['timestamp'].max())
            st.rerun()

    elif state == "working":
        timeseries_all = st.session_state["data"]

        with st.sidebar:
            st.title('📉 Поиск Аномалий во Временных Рядах')

            methods_list = [
                'Isolation Forest',
                'Autoencoder',
                'Multidimensional'
            ]

            numeric_columns = timeseries_all.select_dtypes(include=['float', 'int'])
            numeric_column_names = numeric_columns.columns.tolist()

            selected_method = st.selectbox('Выберите режим поиска аномалий', methods_list)

            if selected_method != 'Multidimensional':
                selected_value = st.selectbox('Выберите столбец для поиска аномалий', numeric_column_names)
                timeseries = pd.concat([timeseries_all['timestamp'], timeseries_all[selected_value]], axis=1)
                timeseries.rename(columns={selected_value: 'value'}, inplace=True)
            else:
                timeseries = timeseries_all

            if st.button("Загрузить другой файл"):
                st.session_state["state"] = "initial"
                st.rerun()

            elif st.button("Анализ временного ряда"):
                st.session_state["state"] = "analyse"
                st.rerun()

            # with st.expander('Подробнее о режимах', expanded=True):
            #     st.write('''
            #         - Текст.
            #         - :orange[**Еще текст**]: про что-то
            #         - :orange[**И еще текст**]: бредовый
            #         ''')

        if selected_method != 'Multidimensional':
            col = st.columns((1.5, 4.5, 2), gap='medium')

            anomaly_df = pd.DataFrame()

            with col[1]:
                st.markdown('#### Обнаруженные аномалии')
                start_datetime = st.session_state["start"]
                end_datetime = st.session_state["end"]
                anomaly_df = find_anomalies(selected_method, timeseries, start_datetime, end_datetime)

            with col[2]:
                st.markdown('#### Аномальные моменты')
                st.dataframe(anomaly_df, width=500)

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

            min_date = pd.to_datetime(timeseries_all['timestamp'].min())
            max_date = pd.to_datetime(timeseries_all['timestamp'].max())

            start_date = st.date_input('Выберите начальную дату', min_value=min_date,
                                       max_value=max_date, value=min_date)
            end_date = st.date_input('Выберите конечную дату', min_value=min_date, max_value=max_date, value=max_date)

            start_datetime = pd.to_datetime(start_date)
            end_datetime = pd.to_datetime(end_date)

            if st.button("Применить фильтр"):
                st.session_state["state"] = "working"
                st.session_state["start"] = start_datetime
                st.session_state["end"] = end_datetime
                st.rerun()

        else:
            pass

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

            # with st.expander('Подробнее о режимах', expanded=True):
            #     st.write('''
            #         - Текст.
            #         - :orange[**Еще текст**]: про что-то
            #         - :orange[**И еще текст**]: бредовый
            #         ''')

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

        elif selected_method == 'Анализ причинно-следственных связей':
            pass


if __name__ == "__main__":
    main()
