from os.path import exists
import webbrowser
import boto3
import pandas as pd
import os
import shap as sh
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PyPDF2 import PdfFileMerger
from fpdf import FPDF
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from sql_aws_connection import aws_connection, specific_query_features_sql


class PDF(FPDF):

    def titles(self, title, y, font_size):
        self.set_xy(0.0, y)
        self.set_font('Arial', 'B', font_size)
        self.set_text_color(150)
        self.cell(w=210.0, h=50.0, align='C', txt=title, border=0)

    def texts(self, text, y, font_size, line_space):
        self.set_xy(33, y)
        self.set_font('Arial', '', font_size)
        # Output justified text
        self.multi_cell(150, 10, txt=text)
        # Line break
        self.ln()
        self.set_text_color(160)

    def two_charts_and_text(self, first_plot, second_plot, third_plot, chart_high, text, text_y, model_i,
                            actual_signal, last_week_return):
        global_x = 25.0
        # top text title
        text_top = f'Asset name: {model_i.asset_name}\nModel name: {model_i.model_name}\n' \
                   f'Prediction: {model_i.predicted_shap_signal}\n' \
                   f'Market direction: {actual_signal} ({last_week_return}%)'
        self.set_xy(global_x, 15.0)
        self.set_font('Arial', '', 11)
        self.multi_cell(300, 7, txt=text_top)
        self.set_text_color(110)
        # first chart i.e. decision plot
        self.set_xy(global_x, 50.0)
        self.image(first_plot, link='', type='', w=750 / 5, h=chart_high - 15)
        # second chart i.e. sum_plot
        self.set_xy(global_x, 160.0)
        self.image(second_plot, link='', type='', w=750 / 5, h=chart_high - 15)
        # third chart i.e. table
        self.set_xy(global_x, 280.0)
        self.image(third_plot, link='', type='', w=800 / 5, h=chart_high)
        # text
        self.texts(text=text, y=text_y, font_size=10, line_space=5)


class Model:
    def __init__(self, feature_val: pd.DataFrame):
        self.feature_val = feature_val

    def features_histogram(self, path_out):
        df = self.feature_val[self.feature_val.index.weekday == 4]
        df_without_1_last = df.iloc[:-1]
        last = (df.iloc[-1]).to_frame()
        last = last.T

        for col in df.columns:
            sns.set_style(style='white')
            sns.displot(df_without_1_last[col], kde=True)
            last_val = last[col].iloc[0]
            plt.axvline(last_val, 0, 40, color='#9e003a')
            plt.savefig(f"{path_out}\\{col}.png")
            plt.close()


class ModelShap(Model):
    def __init__(self, feature_val: pd.DataFrame, shap_avg: pd.DataFrame, model_name: str,
                 date_start=None, date_end=None):
        super().__init__(feature_val)
        self.model_name = model_name
        self.shap_avg = shap_avg
        self.shap_avg, self.feature_val = self.fix_spaces_in_names()
        self.feature_val_tf, self.shap_avg_tf = \
            self.chart_feature_contributions_prepare_data(date_start=date_start, date_end=date_end)
        self.total_shap = self.shap_avg.sum(axis=1)
        self.last_shap_total_prediction = list(self.total_shap)[-1]
        self.predicted_shap_signal = from_flout_to_signal(self.last_shap_total_prediction)
        self.asset_name = self.retrieve_asset_name()
        self.shap_last_date = self.shap_avg.index.strftime("%Y-%m-%d")[-1]

    def retrieve_asset_name(self):
        '''
        get asset name from sql matadata
        '''
        if new_models:
            asset_name = self.model_name.split('_')[0]
        else:
            query = f"""SELECT DISTINCT targets.target_name , models.model_name FROM features_model_mm fmm 
                    left join models on models.model_id = fmm.model_id 
                    left join targets on models.target_id = targets.target_id 
                    WHERE models.live = 1 and models.model_name in ('{self.model_name}') """
            asset_and_model_name = specific_query_features_sql(query)
            asset_name = list(asset_and_model_name['target_name'])[0]
        return asset_name

    def create_shap_table(self):
        '''shap_table includes
        recent_shap - last shap value
        chg - change in shap value from last sample
        pct - percentile location on histogram of last shap sample
        zscore - standard deviations from the mean.
                 If a Z-score is 0, it indicates that the data point's score is identical to the mean score.
        feature_pct, mean, std - of feature value
        '''
        shap_series = self.shap_avg.copy()
        shap_series['Total'] = shap_series.sum(axis=1)
        shap_zscore = shap_series.rank(pct=True)
        shap_zscore_cont = (shap_series - shap_series.mean()) / shap_series['Total'].std()
        feature_zscore = self.feature_val.rank(pct=True)
        shap_cng_from_last_week = shap_series.iloc[-1] - shap_series.iloc[-2]
        shap_table = pd.concat([
            shap_series.iloc[-1].copy().rename('recent_shap'),
            shap_cng_from_last_week.rename('chg'),
            shap_zscore.iloc[-1].copy().rename('pct'),
            shap_zscore_cont.iloc[-1].copy().rename('zscore_cont'),
            feature_zscore.iloc[-1].copy().rename('feature_pct'),
            shap_series.mean().rename('mean'),
            shap_series.std().rename('std'),
        ], axis=1).fillna(0)
        # order = shap_table['recent_shap'].abs().sort_values(ascending=False)
        shap_table.index = shap_table.index.str.strip()
        # shap_table = shap_table.reindex(list(order.index))
        headers_list = list(shap_table.columns)
        headers_list.insert(0, "feature")
        shap_table = shap_table.round(4)
        try:
            shap_table.drop(['expected_value', 'class'], axis=0, inplace=True)
        except:
            shap_table.drop(['expected_value', 'y'], axis=0, inplace=True)
        # order the shap table without total and then add the total to the top of the table
        new_index = list(shap_table.index)
        total_index = new_index.index('Total')
        total = shap_table.iloc[total_index, :]
        new_index.pop(new_index.index('Total'))
        feature_order = list(np.sum(np.abs(self.shap_avg_tf), axis=0).sort_values(ascending=False).index)
        shap_table = shap_table.reindex(feature_order)
        # # select top 10
        # shap_table = shap_table.head(10)
        # order the top 10 without abs
        # order2 = shap_table['recent_shap'].sort_values(ascending=False)
        # shap_table = shap_table.reindex(list(order2.index))
        shap_table = shap_table.append(total)
        # pot total in the first row of the table
        new_index = list(shap_table.index)
        new_index.remove('Total')
        new_index.insert(0, 'Total')
        shap_table = shap_table.reindex(new_index)

        feature_name_size = 140
        other_columns_size = 50
        size_list = [other_columns_size] * 9
        size_list.insert(0, feature_name_size)
        fig = go.Figure(data=[go.Table(
            columnorder=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            columnwidth=size_list,
            header=dict(values=headers_list,
                        fill_color='royalblue',
                        font=dict(color='white', size=12),
                        align=['left', 'center']),
            cells=dict(values=[pd.Series(shap_table.index.values),
                               pd.Series(shap_table['recent_shap'].values),
                               pd.Series(shap_table['chg'].values),
                               pd.Series(shap_table['pct'].values),
                               pd.Series(shap_table['zscore_cont'].values),
                               pd.Series(shap_table['feature_pct'].values),
                               pd.Series(shap_table['mean'].values),
                               pd.Series(shap_table['std'].values),
                               ],
                       fill_color='lavender',
                       # line_color='darkslategray',
                       font_size=12,
                       align=['left', 'center']))
        ])
        fig.update_layout(
            autosize=False,
            width=900,
            height=800,
            margin=dict(
                l=50,
                r=50,
                b=10,
                t=10,
                pad=4
            ))
        return fig, shap_table

    def last_date_weekly_shap_decision_plot(self):
        '''returns the last week decision plot'''
        self.shap_avg.index.strftime("%Y-%m-%d")
        shap_avg_one_date = self.shap_avg.iloc[-1].copy()
        base_value = shap_avg_one_date['expected_value']
        shap_avg_one_date.drop(labels=['expected_value'], inplace=True)
        order = shap_avg_one_date.abs().sort_values(ascending=False)
        shap_avg_one_date = shap_avg_one_date[order.index]
        shap_avg_one_date_val = shap_avg_one_date.values
        feature_names_list = shap_avg_one_date.index.to_list()
        feature_names_list_len = np.alen(feature_names_list) + 1
        return base_value, feature_names_list, shap_avg_one_date_val, feature_names_list_len

    def chart_feature_contributions_prepare_data(self, date_start=None, date_end=None):
        '''prepare the data to the time series history plot for each feature, i.e.
        drop expected column from shap df
        drop class column from df_final i.e. feature values df
         select the same dates for both feature values and shap dfs'''
        date_start = pd.Timestamp(date_start or self.shap_avg.index[0])
        date_end = pd.Timestamp(date_end or self.shap_avg.index[-1])
        shap_avg_tf = self.shap_avg[date_start:date_end].drop(['expected_value'], axis=1)
        try:
            feature_val_tf = self.feature_val[date_start:date_end].drop(['class'], axis=1)
        except:
            feature_val_tf = self.feature_val[date_start:date_end].drop(['y'], axis=1)
        feature_val_tf = feature_val_tf[feature_val_tf.index.isin(shap_avg_tf.index)]
        return feature_val_tf, shap_avg_tf

    def chart_feature_contributions_plot(self, i):
        '''returen time series history plot for each feature.
        include : feature shap, feature value and total shap predicted (the prediction)'''
        ax1 = self.shap_avg_tf[i].plot(title=f'Feature_contributions - {i}', figsize=(8, 5), label='Feature Shap',
                                       color='#010fcc')
        ax2 = self.shap_avg_tf.sum(axis=1).plot(color='#010fcc', alpha=0.4, label='Total Shap Predicted')
        ax3 = self.feature_val_tf[i].plot(secondary_y=True, color='#9e003a', alpha=0.6, label='Feature Value')
        ax3.left_ax.axhline(y=0, xmin=0, xmax=1, color='k', alpha=0.35, linestyle='-')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax3.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, frameon=False, loc='upper center', ncol=3)

    def sum_plot_prepare_data(self):
        '''prepare the data for shap summary plot'''
        shap_avg_tf_val = self.shap_avg_tf.values
        feature_names_list_len = np.alen(self.shap_avg_tf.columns.to_list())
        return shap_avg_tf_val, feature_names_list_len

    def fix_spaces_in_names(self):
        '''correct spaces in features name in crucial for the clickable html table to work'''
        for df in [self.shap_avg, self.feature_val]:
            df.columns = [c.replace(' ', '_') for c in df.columns]
        return self.shap_avg, self.feature_val


def write_a_pdf(text_dict, chart_high, path_out, text_y, model_i, actual_signal, last_week_return):
    '''create a pdf to each sub model using 2 plots, 1 table and text.'''
    pdf = PDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    # chart
    first_plot = (fr"{path_out}\decision_plot_{model_i.model_name}.png")
    second_plot = (fr"{path_out}\sum_plot_{model_i.model_name}.png")
    third_plot = (fr"{path_out}\table_{model_i.model_name}.png")
    text = f'Main Insights: {text_dict[model_i.model_name][0]}'
    text = text.replace(r'\n', '\n')
    pdf.two_charts_and_text(first_plot, second_plot, third_plot, chart_high, text, text_y=text_y, model_i=model_i,
                            actual_signal=actual_signal, last_week_return=last_week_return)
    pdf.output(fr'{path_out}\{model_i.model_name}.pdf', 'F')
    pdf.close()


def create_models_pdf(chart_high_global, text_y, path_out, model_i, actual_signal, last_week_return, text_dict):
    '''insert the data needed to use write_a_pdf function.
    the function split between AUD model that have a lot of features and need more space
    to the rest of the models'''
    if model_i.model_name == 'TD_LGBM_CAT_AUD':
        chart_high = chart_high_global
        text_y = text_y * 2
        write_a_pdf(text_dict,
                    chart_high=chart_high, path_out=path_out, text_y=text_y,
                    model_i=model_i, actual_signal=actual_signal, last_week_return=last_week_return)
    else:
        chart_high = chart_high_global / 2 + 30
        text_y = text_y
        write_a_pdf(text_dict,
                    chart_high=chart_high, path_out=path_out, text_y=text_y,
                    model_i=model_i, actual_signal=actual_signal, last_week_return=last_week_return)


def merge_pdfs(pdfs, path_out, shap_last_date):
    '''take all models pdf and combine them'''
    merger = PdfFileMerger()
    shap_last_date = shap_last_date.replace('-', '')
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(f'{path_out}\\Weekly_SHAP_{shap_last_date}.pdf')
    merger.close()


def get_weekly_returns(main_folder):
    '''take weekly returens from aws s3 and correct it so it will match log(transformed_data)'''
    weekly_returns = pd.read_csv(f'{main_folder}\weekly_returns.csv')
    weekly_returns = (np.exp(weekly_returns)).cumprod().pct_change()
    return weekly_returns


def s3(bucket_name, sub_folder, limit, **kwargs):
    '''connect to s3'''
    return boto3.client('s3').list_objects_v2(Bucket=bucket_name, Prefix=sub_folder, MaxKeys=limit, **kwargs)


def get_models_input_from_s3(model):
    '''take data from eval on s3'''
    bucket = 'fart'
    res = s3(bucket, fr'{models_path_in[model]}', limit=300)
    for i in res.get('Contents', []):
        if 'df_final_td.csv' in i.get('Key'):
            df_final = aws_connection(Bucket=bucket, Key=i.get('Key'))
    key = fr'{models_path_in[model]}/{model}_mean_shap_values.csv'
    shap_avg = aws_connection(Bucket=bucket, Key=key)
    return df_final, shap_avg


def change_transformed_name_to_model_name(model):
    '''change targets transformed data name to production target name '''
    query = f"""SELECT DISTINCT targets.transformed_name , models.model_name FROM features_model_mm fmm 
            left join models on models.model_id = fmm.model_id 
            left join targets on models.target_id = targets.target_id 
            WHERE models.live = 1 and models.model_name in ('{model}') """
    tr_name = specific_query_features_sql(query)
    return list(tr_name['transformed_name'])


def from_flout_to_signal(flout):
    '''this function takes a flout and returns the signal : short/long/flat'''
    f = lambda x: 'Long' if x > 0 else 'Short' if x < 0 else 'Flat' if x == 0 else np.nan
    signal = f(flout)
    return signal


def after_pred_realized_create_data(model_i, weekly_returns):
    ''' return :
    hit_miss -
    actual_signal - Long/Short and
    last_week_return - market weekly return

    use:
    shap total value
    weekly return'''
    tr_name = change_transformed_name_to_model_name(model_i.model_name)
    last_week_return = np.round(weekly_returns[f'{tr_name[0]}'].iloc[-1] * 100, 1)
    actual_signal = from_flout_to_signal(last_week_return)
    '''hit'''
    f = lambda x, y: 'Hit' if (((x > 0) and (y > 0)) or ((x < 0) and (y < 0))) else 'Miss'
    hit_miss = f(last_week_return, model_i.last_shap_total_prediction)
    return hit_miss, actual_signal, last_week_return


def decision_plot_as_png(path_out, model_i):
    '''return decision plot of the last sample
     use:
     base_value- the average value of the history shap if there was no features in the model, i.e. the intercept
     feature_names_list- model features list
     shap_avg_one_date_val- shap value of the last sample
     feature_names_list_len- length of the model name list
     '''
    base_value, feature_names_list, shap_avg_one_date_val, feature_names_list_len = \
        model_i.last_date_weekly_shap_decision_plot()
    sh.decision_plot(base_value, shap_avg_one_date_val, feature_names=feature_names_list, title=f'Decision Plot',
                     show=False, feature_display_range=slice(-1, -feature_names_list_len, -1))
    plt.savefig(fr'{path_out}\decision_plot_{model_i.model_name}.png', bbox_inches='tight')
    plt.close()


def create_sum_plot(path_out, model_i):
    '''return 1 png of shap summary plot'''
    shap_avg_tf_val, feature_names_list_len = model_i.sum_plot_prepare_data()
    sh.summary_plot(shap_avg_tf_val, model_i.feature_val_tf, max_display=feature_names_list_len, show=False)
    plt.title('Summary Plot')
    plt.savefig(f"{path_out}\\sum_plot_{model_i.model_name}.png", bbox_inches='tight')
    plt.close()


def create_time_series_plots(history_plot_path, model_i):
    '''return folder of png time-series history plot to each feature in model'''
    '''open folders'''
    folder = fr'{history_plot_path}\history_plot\{model_i.model_name}'
    open_folder(folder)
    '''create charts'''
    for i in model_i.shap_avg_tf:
        model_i.chart_feature_contributions_plot(i)
        plt.savefig(fr'{folder}\{model_i.model_name}_{i}.png', bbox_inches='tight')
        plt.close()


def create_html_to_each_history_plot(model_i, path_out, history_plot_path):
    '''opens html of history time series plot and histogram plot for each feature'''
    folder = fr'{path_out}/history_plot_html/{model_i.model_name}'
    open_folder(folder)
    for feature in model_i.shap_avg_tf:
        f = open(rf'{folder}/{feature}.html', 'w')
        history_plot = fr"{history_plot_path}/history_plot/{model_i.model_name}/{model_i.model_name}_{feature}.png"
        histogram_plot = fr"{path_out}/histogram/{model_i.model_name}/{feature}.png"
        message = f'''
                 <html>
                     <body>
                     <div class="row">
                         <div class="column">
                           <img src={history_plot}>
                         </div>
                         <div class="column">
                           <img src={histogram_plot}>
                         </div>
                     </div>
                     </body>
                 </html>
                 '''
        f.write(message)
        f.close()


def create_pred_status(weekly_returns_last_date, model_i, weekly_returns):
    '''returen
    hit_miss -
    actual_signal - Long/Short and
    last_week_return - market weekly return
    for each prediction status: before and after prediction realized'''
    '''create data After prediction realized'''
    # if we are after prediction date, create actual signal , hit , weekly return
    if weekly_returns_last_date == model_i.shap_last_date:
        pred_status = 'After prediction realized'
        hit_miss, actual_signal, last_week_return = \
            after_pred_realized_create_data(model_i, weekly_returns)
    else:
        pred_status = 'Before prediction realized'
        hit_miss = '?'
        actual_signal = '?'
        last_week_return = '?'
    return pred_status, hit_miss, actual_signal, last_week_return


def create_all_tables_text_and_charts(model_i, weekly_returns_last_date, weekly_returns):
    '''return:
    table
    decision_plot
    sum_plot
    history_shap - if history_shap True in main the the code will create the plots for each feature and locate it in
                    W:\Macrobot\Analysis\SHAP\weekly_shap\SHAP_2017\history_plot
                   if history_shap False in main the the code will take the shap from the same folder:
                   W:\Macrobot\Analysis\SHAP\weekly_shap\SHAP_2017\history_plot
    histogram_shap - histogram plot to each feature
    create pred_status, hit_miss, actual_signal, last_week_return after/before prediction realized
    '''
    '''create tables as png'''
    fig, shap_table = model_i.create_shap_table()
    fig.write_image(fr'{path_out}\table_{model_i.model_name}.png', scale=1)
    '''create decision plot as png'''
    print("creating: decision_plot")
    decision_plot_as_png(path_out, model_i)
    '''create summary plot'''
    print("creating: sum_plot")
    create_sum_plot(path_out, model_i)
    '''feature history shap'''
    if history_shap:
        print("creating: history_shap")
        create_time_series_plots(history_plot_path, model_i)
    '''create histograms'''
    print("creating: histogram_shap")
    hist_path_out = fr'{path_out}\histogram\{model_i.model_name}'
    open_folder(hist_path_out)
    model_i.features_histogram(hist_path_out)
    '''create data after/before prediction realized'''
    print("creating: create data after/before prediction realized")
    pred_status, hit_miss, actual_signal, last_week_return = \
        create_pred_status(weekly_returns_last_date, model_i, weekly_returns)
    return pred_status, hit_miss, actual_signal, last_week_return, shap_table
    print(" ")


def create_cover_page(shap_last_date, assets, market_text):
    '''return pdf cover page with title, prediction date, assets in this analysis'''
    pdf = PDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.titles(title='Weekly Prediction Analysis', y=5.0, font_size=28)
    pdf.titles(title=f'Prediction date for: {shap_last_date}', y=20.0, font_size=18)
    pdf.titles(title=f'Assets in this analysis: {set(assets)}', y=30.0, font_size=14)
    pdf.texts(text=market_text, y=70, font_size=12, line_space=10)
    pdf.output(fr'{path_out}\cover_page.pdf', 'F')
    pdf.close()


def color_column_value(val):
    '''style for table'''
    GREEN = 'SeaGreen'
    RED = 'IndianRed'
    NORMAL = 'black'
    color = NORMAL
    if isinstance(val, (float, int)):
        if val > 1e-6:  # val > 0
            color = GREEN
        elif val < -1e-6:  # val < 0
            color = RED

    elif isinstance(val, (bool)):
        color = GREEN if val else RED
    elif isinstance(val, (str)):
        if val.lower() in ['buy', 'long']:
            color = GREEN
        elif val.lower() in ['sell', 'short']:
            color = RED

    return 'color: %s' % color


def set_default_style(df, color_columns=None, format=None):
    '''default style for table'''
    style = df.style.set_properties(
        **dict([
            ('background-color', 'White'),
            ('font-family',
             "Helvetica Neue,Helvetica,PingFang SC,Hiragino Sans GB,Microsoft YaHei,Arial,sans-serif"),
            ('font-size', '12px'),
            ('color', 'Black'),
            ('margin', '0'),
            ('margin-left', '100px'),
            ('vertical-align', 'top'),
            ('text-align', 'right'),
            ('text-align', '50px'),
            ('box-sizing', 'border-box'),
            ('border-collapse', 'collapse'),
            ('border', '1px solid rgb(232, 234, 236)'),
            ('padding', '5px'),
            ('align', 'center')
        ]))
    style = style.set_table_styles([
        dict(selector='th', props=[
            ('background-color', 'WHITESMOKE'),
            ('border', '1px solid rgb(232, 234, 236)'),
            ('border-collapse', 'collapse'),
            ('box-sizing:', 'border-box'),
            ('text-align', 'center'),
            ('vertical-align', 'center'),
            ('color', 'Black'),
            ('font-family', "Helvetica Neue,Helvetica,PingFang SC,Hiragino Sans GB,Microsoft YaHei,Arial,sans-serif"),
            ('font-size', '12px'),
            ('padding', '5px'),
            ('align', 'center')
        ]),
    ])

    if color_columns and isinstance(color_columns, list):
        for i in color_columns:
            style = style.applymap(color_column_value(), subset=i)

    if isinstance(format, dict):
        style = style.format(format)
    elif isinstance(format, list):  # expecting tuples in list, [ ( idx, format ) ]
        for idx, formatter in format:
            style = style.format(formatter, subset=idx)

    return style.set_na_rep('-')


def table_style(df):
    '''color table so that high std gets stronger blue color
    pct get blue color for smaller change and pink for bigger change
    chg get blue color for negative values and pink for positive
    '''
    cm_value = sns.light_palette("royalblue", as_cmap=True)
    cm_signs = sns.diverging_palette(260, 340, n=3, center='light', as_cmap=True)
    style = set_default_style(df).format('{:.4f}')
    for i in ['std']:
        style = style.background_gradient(cmap=cm_value,
                                          subset=pd.IndexSlice[df.drop(labels='Total').index, i],
                                          # gmap=df[i].abs()
                                          )
    for i in ['pct']:
        vcenter = 0 if i == 'next_week' else 0.5
        # norm = TwoSlopeNorm(vmin=min(df[i].min(), -1e-9), vmax=max(df[i].max(), 1e-9), vcenter=vcenter)
        style = style.background_gradient(cmap=cm_signs,
                                          subset=pd.IndexSlice[df.drop(labels='Total').index, i],
                                          # gmap=norm(df[i].drop(labels='Total'))
                                          )

    for i in ['chg']:
        # norm = TwoSlopeNorm(vmin=min(df[i].min(), -1e-9), vmax=max(df[i].max(), 1e-9), vcenter=0)
        style = style.background_gradient(cmap=cm_signs,
                                          subset=pd.IndexSlice[df.drop(labels='Total').index, i],
                                          # gmap=norm(df[i].drop(labels='Total'))
                                          )
    return f'<div>{style.render()}</div>'


def clickable_table_html(shap_table, model_i, path_out):
    ''' return styled clickable table that take from the main html
    to a specific feature html by clicking at the feature name in the first column of the table
    target=_blank means that after clicking on feature a new html path is open (and not change the current main html)
    take the path folder where all htmls are located'''
    path = rf'{path_out}/history_plot_html'
    for feature in range(1, shap_table.index.value_counts().sum()):
        link_name = rf'<a target=_blank href="file:///{path}/{model_i.model_name}/{shap_table.index[feature]}.html">' \
                    rf'{shap_table.index[feature]}</a>'
        shap_table = shap_table.rename(index={f'{shap_table.index[feature]}': f'{link_name}'})
    style = table_style(shap_table)
    return style


def create_html_input_list(model_i, text_dict, html_table, actual_signal, last_week_return):
    '''return string of all images and text in html'''
    html_input_list = []
    first_plot = (fr"{path_out}\decision_plot_{model_i.model_name}.png")
    second_plot = (fr"{path_out}\sum_plot_{model_i.model_name}.png")
    text = f'Main Insights: {text_dict[model_i.model_name][0]}'
    text = text.replace(r'\n', '\n')
    png_model_list = [first_plot, second_plot]
    upper_text_li = [f'Asset name: {model_i.asset_name}', f'Model name: {model_i.model_name}',
                     f'Prediction: {model_i.predicted_shap_signal}',
                     f'Market direction: {actual_signal} ({last_week_return}%)']
    html_input_list.append(f'<div class="model_text">')
    for li in upper_text_li:
        html_input_list.append(f'<h3><pre>{li}</pre></h3>')
    html_input_list.append(f'</div>')
    html_input_list.append(f'<div class="feature_text"><h3><pre class="cut_text_lines">{text}</pre></h3></div>')
    for fig in png_model_list:
        html_input_list.append(f'<div class="imgs"><img src={fig}></img></div>')
    html_input_list.append(f'<div class="tables">{html_table}</div>')
    return html_input_list


def create_model_input_html(path_out, model_i, shap_table, text_dict, history_plot_path, actual_signal,
                            last_week_return):
    '''
    return string input for html includes:
    history plot fore each feature html
    clickable html table
    text and images html strings
    '''
    create_html_to_each_history_plot(model_i, path_out, history_plot_path)
    html_table = clickable_table_html(shap_table, model_i, path_out)
    html_input_list_per_model = create_html_input_list(model_i, text_dict, html_table, actual_signal, last_week_return)
    return html_input_list_per_model


def open_folder(folder):
    '''open html folder'''
    if not os.path.exists(folder):
        os.makedirs(folder)


def create_final_html(html_input_list_for_all_models, path_out, shap_last_date, assets, market_text, status):
    '''take list of all pngs and text from all models and create a final html.
    the container includes all css styles'''
    html_name = fr'{path_out}/{status}_final_{shap_last_date}.html'
    f = open(html_name, 'w')
    css = '''
                .container {
                    grid-template-columns: [line1] 10% [line2] 80% [line3] 10% [end];
                    grid-template-rows: [row1] 15%;
                    row-gap: 15px;
                    margin-top: 10%;
                    margin-right: 20%;
                    margin-bottom: 20%;
                    margin-left: 10%;
                }
                .titles {
                margin-bottom: 6%;
                }
                .market_text {

                }
                .model_text {  
                padding: 6%;          
                }
                .feature_text {
                }

                .imgs {
                margin-top: 10%;
                margin-right: 10%;
                margin-left: 10%;
                margin-bottom: 5%;    
                }  

                .tables {
                margin-left: 25%;

                }

                .cut_text_lines {
                word-break: break-word;
                width: 500px;
                white-space:break-spaces;
                }
                '''

    message = f'''
            <html>
                <head>
                    <style>
                            {css}
                    </style>
                </head>
                <body>
                    <section class="container">
                        <div class="titles">
                            <h1><pre style="font-size: 36px"> Weekly Prediction Analysis</h1>
                            <h2><pre> Status: {status}</h2>
                            <h2><pre> Prediction date for: {shap_last_date}</h2>
                            <h3><pre> Production folder: {key}{pre_prediction_date}</h3>
                            <h3><pre> Assets in this analysis: {set(assets)}\n\n\n</h3>
                        </div>
                        <div class="market_text">
                            <h3><pre class="cut_text_lines">This week: {market_text}</pre></h3>
                        </div>
                        <div>
                        {' '.join(html_input_list_for_all_models)}
                        </div>
                    </section>
                </body>
            </html>
            '''
    f.write(message)
    f.close()
    webbrowser.open_new_tab(html_name)


def get_history_shap(model, history_shap_path):
    ''' get the history of shap values'''
    shap_avg_history = pd.read_csv(fr'{history_shap_path}\{model}_mean_shap_values.csv',
                                   index_col=[0], parse_dates=[0], dayfirst=True)
    return shap_avg_history


def merge_prod_shap_with_history(new_shap_avg, shap_avg_history):
    if new_shap_avg.index[-1] == shap_avg_history.index[-1]:
        shap_avg = shap_avg_history
    else:
        found_index = False
        if new_shap_avg.index[-1] - shap_avg_history.index[-1] == timedelta(days=7):
            shap_avg_history = shap_avg_history.append(new_shap_avg.iloc[-1, :])
            found_index = True
        if not found_index:
            i = 1
            while new_shap_avg.index[-i] != shap_avg_history.index[-1]:
                i += 1
            for j in range(i - 1, 0, -1):
                shap_avg_history = shap_avg_history.append(new_shap_avg.iloc[-j, :])
        shap_avg = shap_avg_history
    return shap_avg


def loop_on_models(models_path_in, weekly_returns_last_date, weekly_returns, text_dict, status, history_plot_path):
    '''
    return html and pdf for each model and create final pdf and final html combine all models
    loop on all sub models that are in models_path_in dictionary
    '''
    pdfs = []
    assets = []
    html_input_list_for_all_models = []
    for model in models_path_in.keys():
        print(f"\nmodel: {model}")
        '''A.1. get the data'''
        # get new production shap, df_final
        df_final, new_shap_avg = get_models_input_from_s3(model)
        if new_models:
            shap_avg = new_shap_avg
        else:
            # get history shap
            shap_avg_history = get_history_shap(model, history_plot_path)
            # merge shap history with new and save
            shap_avg = merge_prod_shap_with_history(new_shap_avg, shap_avg_history)
            shap_avg.to_csv(rf'{history_plot_path}\{model}_mean_shap_values.csv', date_format='%Y-%m-%d')
        if status == 'post_prediction':
            shap_avg = shap_avg.iloc[:-1, :]
            df_final = df_final.iloc[:-1, :]
        '''A.2. start a ModelShap Class'''
        model_i = ModelShap(feature_val=df_final, shap_avg=shap_avg, model_name=model)
        '''A.3. append asset name to a list of assets'''
        assets.append(model_i.asset_name)
        '''B. create the charts and tables'''
        pred_status, hit_miss, actual_signal, \
        last_week_return, shap_table = create_all_tables_text_and_charts(model_i, weekly_returns_last_date,
                                                                         weekly_returns)
        '''C. create html'''
        print(f"creating: html")
        html_input_list_per_model = create_model_input_html(path_out, model_i, shap_table, text_dict, history_plot_path,
                                                            actual_signal, last_week_return)
        html_input_list_for_all_models.extend(html_input_list_per_model)

        '''D. create pdf for each model'''
        # print(f"creating: pdf")
        # pdfs.append(fr"{path_out}\{model_i.model_name}.pdf")
        # create_models_pdf(chart_high_global=170, text_y=50, path_out=path_out, model_i=model_i,
        #                   actual_signal=actual_signal, last_week_return=last_week_return, text_dict=text_dict)

    '''E. create a cover page'''
    # print(f"creating: cover_page")
    market_text = text_dict['market'][0]
    market_text = market_text.replace(r'\n', '\n')
    # create_cover_page(model_i.shap_last_date, assets, market_text)
    # pdfs.insert(0, fr"{path_out}\cover_page.pdf")

    '''F. merge all models pdfs to one '''
    # print(f"creating: final pdf")
    # merge_pdfs(pdfs, path_out, model_i.shap_last_date)

    '''G. create a final HTML'''
    print(f"creating: final html")
    create_final_html(html_input_list_for_all_models, path_out, model_i.shap_last_date, assets, market_text, status)


def get_general_data(main_folder):
    '''this function get all non specific model data, i.e. weekly returns and text for all models'''
    '''get last weekly returns date'''
    weekly_returns = get_weekly_returns(main_folder)
    weekly_returns_last_date = weekly_returns.index.strftime("%Y-%m-%d")[-1]
    '''get text data'''
    if status == 'post_prediction':
        date = rf'{post_prediction_date + timedelta(7)}'
        open_text_csv_if_not_exists(path_out, date)
        text_data = pd.read_csv(rf'{path_out}\text_file_{date}.csv', index_col=[0], encoding='cp1252')
        text_dict = text_data.T.to_dict('list')
    if status == 'pre_prediction':
        date = rf'{pre_prediction_date + timedelta(7)}'
        open_text_csv_if_not_exists(path_out, date)
        text_data = pd.read_csv(rf'{path_out}\text_file_{date}.csv', index_col=[0])
        text_dict = text_data.T.to_dict('list')
    return weekly_returns, weekly_returns_last_date, text_dict


def get_friday_date(last=True):
    '''give the last friday date if last=True,
    or the friday before is last=False'''
    day = date.today()
    while day.weekday() != 4:
        day -= timedelta(1)
    if last:
        friday = day
    else:
        day -= timedelta(7)
        friday = day
    return friday


def open_output_folder(status, current_time_and_date, main_folder):
    '''open output folder '''
    if a_try:
        folder = fr'{main_folder}\weekly_shap_{current_time_and_date}--TRY\{status}'
    else:
        folder = fr'{main_folder}\weekly_shap_{pre_prediction_date}\{status}'
    open_folder(folder)
    return folder


def get_production_date():
    '''return fridays date for the last two weeks'''
    post_prediction_date = get_friday_date(last=False)
    pre_prediction_date = get_friday_date(last=True)
    return post_prediction_date, pre_prediction_date


def open_text_csv_if_not_exists(path_out, date):
    '''return generic csv text file that have model name and generic text for each model
     if text csv is not exist-
     after the fist creation analyst should add their incites to each model text box'''
    path_to_text_file = rf'{path_out}\text_file_{date}.csv'
    generic_text_dict = {}
    for model in models_path_in:
        generic_text_dict[f'market'] = [r'text.. text... text ... text.. text... text ... '
                                        r'text.. text... text ... text.. text... text ...\n']
        generic_text_dict[f'{model}'] = [r'text.. text... text ... text.. text... text ... '
                                         r'text.. text... text ... text.. text... text ...\n']
    if not exists(path_to_text_file):
        text_df = pd.DataFrame.from_dict(generic_text_dict, orient='index', columns=['text'])
        text_df.to_csv(fr'{path_to_text_file}')


def main(models_path_in, status, main_folder):
    '''create and open pre and post prediction html and pdf for all models in models_path_in dictionary'''
    weekly_returns, weekly_returns_last_date, text_dict = get_general_data(main_folder)
    print(f'status: {status}')
    loop_on_models(models_path_in, weekly_returns_last_date, weekly_returns, text_dict, status, history_plot_path)


if __name__ == '__main__':
    ''' 
    returns for each pre and post prediction status a final html and pdf for all sub model include in the analysis
    parameters:
    a_try- if True the code adds to main output folder the current date and hour to avoid running over production output
    bucket- the s3 bucket 'fart-eval'
    key- the s3 key i.e. folder 'production_' or 'simulation_'
    history_shap- if True in main the the code will create the plots for each feature and locate it in history_plot_path
                   if False in main the the code will take the shap from the same folder.
    history_plot_path- path to history output folder fr'W:\Macrobot\Analysis\SHAP\weekly_shap\SHAP_2017'
    models_path_in- dict that contain all sub models name and path from s3
                    for example  models_path_in = {'TD_LGBM_REG_SPX': rf'{key}{pre_prediction_date}/SNP500_0/output'}
    '''
    a_try = True
    new_models = True
    current_time_and_date = datetime.now().strftime("%Y-%m-%d_%H")
    bucket = 'bucket_name'
    key = r'key_name'
    history_shap = True
    history_plot_path = fr'W:\Macrobot\Analysis\SHAP\weekly_shap\SHAP_2017'
    post_prediction_date, pre_prediction_date = get_production_date()
    models_path_in = {
        'TD_LGBM_REG_SPX': rf'{history_plot_path}',
        'TD_LGBM_CAT_AUD': rf'{history_plot_path}',
        'US30Y': rf'{history_plot_path}',
    }
    main_folder = r'P:\MB\SHAP\new'
    for status in ['post_prediction', 'pre_prediction']:
        path_out = open_output_folder(status, current_time_and_date, main_folder)
        main(models_path_in, status=status, main_folder=main_folder)

