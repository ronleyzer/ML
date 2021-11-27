# Load Matplotlib and data wrangling libraries.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

def hist():
    '''histograms'''
    param_list = ['gym_distance_from_home', 'gym_distance_from_work', 'month', 'duration', 'start_hour']
    for weight in ['customer_weight', 'visit_weight', 'customer_and_visit_weight']:
        '''open folders to each weight option'''
        folder = fr'{path_in}\out\{weight}'
        open_folder(folder)

        '''cumulative quantiles'''
        for param in param_list:
            for key in gym_data.keys():
                gym_data[weight][key][param+'_rank'] = gym_data[weight][key][param].rank(pct=True)
                gym_data[weight][key].sort_values(by=f'{param}_rank', inplace=True)
                w_mean = np.average(gym_data[weight][key][param], weights=gym_data[weight][key][weight])
                plt.plot(gym_data[weight][key][f'{param}_rank'], gym_data[weight][key][f'{param}'], label=f'{key}, w_mean: {np.round(w_mean,1)}')
            plt.legend()
            plt.title(f"{param} weight: {weight}")
            plt.savefig(fr"{folder}\{weight}_cumulative_distribution_{param}.png")
            plt.close()

        '''histograms'''
        for param in param_list:
            for key in gym_data.keys():
                df = gym_data[weight][key].copy()
                if param == 'gym_distance_from_work':
                    df = df.dropna()
                df = df[df[f'{param}_rank'] < 0.94]
                w_mean = np.average(df[param], weights=df[weight])
                plt.hist(df[param], label=f'{key}, w_mean: {np.round(w_mean,1)}', histtype="step", density=True, bins=50, weights=df[weight])
            plt.legend()
            plt.title(f"{param} weight: {weight}")
            plt.savefig(fr"{folder}\{weight}_hist_{param}.png")
            plt.close()

            for key in number_of_visits_per_costumer.keys():
                w_mean = np.average(number_of_visits_per_costumer[key][f'number_of_visit'],
                                          weights=number_of_visits_per_costumer[key][weight])
                plt.hist(number_of_visits_per_costumer[key]['number_of_visit'], label=f'{key}, w_mean: {np.round(w_mean,1)}', histtype="step", density=True,
                         bins=50, weights=number_of_visits_per_costumer[key][f'{weight}'])
            plt.legend()
            plt.title(f"number of visits per costumer weight: {weight}")
            plt.savefig(fr"{folder}\{weight}_hist_gym_number_of_visits_per_costumer_{param}.png")
            plt.close()


def bar_plot(df, feature):
    # Use Matplotlib's font manager to rebuild the font library.
    mpl.font_manager._rebuild()

    # Use the newly integrated Roboto font family for all text.
    plt.rc('font', family='Open Sans')

    fig, ax = plt.subplots()

    # Save the chart so we can loop through the bars below.

    bars = ax.bar(
        x=np.arange(df.size),
        height=df[feature],
        tick_label=df.index
    )

    # Axis formatting.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    # Add text annotations to the top of the bars.
    bar_color = bars[0].get_facecolor()
    for bar in bars:
      ax.text(
          bar.get_x() + bar.get_width() / 2,
          bar.get_height() + 0.3,
          round(bar.get_height(), 1),
          horizontalalignment='center',
          color=bar_color,
          weight='bold'
      )

    # Add labels and a title.
    ax.set_xlabel('Year of Car Release', labelpad=15, color='#333333')
    ax.set_ylabel('Average Miles per Gallon (mpg)', labelpad=15, color='#333333')
    ax.set_title('Average MPG in Cars [1970-1982]', pad=15, color='#333333',
                 weight='bold')

    fig.tight_layout()
    plt.show()

