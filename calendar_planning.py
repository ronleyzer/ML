# import calendar
# # Create a plain text calendar
# c = calendar.TextCalendar(calendar.THURSDAY)
# str = c.formatmonth(2025, 1, 0, 0)
# print (str)
#
# # Create an HTML formatted calendar
# hc = calendar.HTMLCalendar(calendar.THURSDAY)
# str = hc.formatmonth(2025, 1)
# print (str)
# print("WTF")
import calendar
# yio = int(input('Input a month'))
# s = int(input('Input a year'))
# print(calendar.month(s, yio))
import webbrowser
from tkcalendar import Calendar
from datetime import datetime


class Agenda(Calendar):

    def __init__(self, master=None, **kw):
        Calendar.__init__(self, master, **kw)
        # change a bit the options of the labels to improve display
        for i, row in enumerate(self._calendar):
            for j, label in enumerate(row):
                self._cal_frame.rowconfigure(i + 1, uniform=1)
                self._cal_frame.columnconfigure(j + 1, uniform=1)
                label.configure(justify="center", anchor="n", padding=(1, 4))
                self['firstweekday'] = 'sunday'

    def _display_days_without_othermonthdays(self):
        year, month = self._date.year, self._date.month

        cal = self._cal.monthdays2calendar(year, month)
        while len(cal) < 6:
            cal.append([(0, i) for i in range(7)])

        week_days = {i: 'normal.%s.TLabel' % self._style_prefixe for i in
                     range(7)}  # style names depending on the type of day
        week_days[self['weekenddays'][0] - 1] = 'we.%s.TLabel' % self._style_prefixe
        week_days[self['weekenddays'][1] - 1] = 'we.%s.TLabel' % self._style_prefixe
        _, week_nb, d = self._date.isocalendar()
        if d == 7 and self['firstweekday'] == 'sunday':
            week_nb += 1
        modulo = max(week_nb, 52)
        for i_week in range(6):
            if i_week == 0 or cal[i_week][0][0]:
                self._week_nbs[i_week].configure(text=str((week_nb + i_week - 1) % modulo + 1))
            else:
                self._week_nbs[i_week].configure(text='')
            for i_day in range(7):
                day_number, week_day = cal[i_week][i_day]
                style = week_days[i_day]
                label = self._calendar[i_week][i_day]
                label.state(['!disabled'])
                if day_number:
                    txt = str(day_number)
                    label.configure(text=txt, style=style)
                    date = self.date(year, month, day_number)
                    if date in self._calevent_dates:
                        ev_ids = self._calevent_dates[date]
                        i = len(ev_ids) - 1
                        while i >= 0 and not self.calevents[ev_ids[i]]['tags']:
                            i -= 1
                        if i >= 0:
                            tag = self.calevents[ev_ids[i]]['tags'][-1]
                            label.configure(style='tag_%s.%s.TLabel' % (tag, self._style_prefixe))
                        # modified lines:
                        text = '%s\n' % day_number + '\n'.join([self.calevents[ev]['text'] for ev in ev_ids])
                        label.configure(text=text)
                else:
                    label.configure(text='', style=style)

    def _display_days_with_othermonthdays(self):
        year, month = self._date.year, self._date.month

        cal = self._cal.monthdatescalendar(year, month)

        next_m = month + 1
        y = year
        if next_m == 13:
            next_m = 1
            y += 1
        if len(cal) < 6:
            if cal[-1][-1].month == month:
                i = 0
            else:
                i = 1
            cal.append(self._cal.monthdatescalendar(y, next_m)[i])
            if len(cal) < 6:
                cal.append(self._cal.monthdatescalendar(y, next_m)[i + 1])

        week_days = {i: 'normal' for i in range(7)}  # style names depending on the type of day
        week_days[self['weekenddays'][0] - 1] = 'we'
        week_days[self['weekenddays'][1] - 1] = 'we'
        prev_m = (month - 2) % 12 + 1
        months = {month: '.%s.TLabel' % self._style_prefixe,
                  next_m: '_om.%s.TLabel' % self._style_prefixe,
                  prev_m: '_om.%s.TLabel' % self._style_prefixe}

        week_nb = cal[0][1].isocalendar()[1]
        modulo = max(week_nb, 52)
        for i_week in range(6):
            self._week_nbs[i_week].configure(text=str((week_nb + i_week - 1) % modulo + 1))
            for i_day in range(7):
                style = week_days[i_day] + months[cal[i_week][i_day].month]
                label = self._calendar[i_week][i_day]
                label.state(['!disabled'])
                txt = str(cal[i_week][i_day].day)
                label.configure(text=txt, style=style)
                if cal[i_week][i_day] in self._calevent_dates:
                    date = cal[i_week][i_day]
                    ev_ids = self._calevent_dates[date]
                    i = len(ev_ids) - 1
                    while i >= 0 and not self.calevents[ev_ids[i]]['tags']:
                        i -= 1
                    if i >= 0:
                        tag = self.calevents[ev_ids[i]]['tags'][-1]
                        label.configure(style='tag_%s.%s.TLabel' % (tag, self._style_prefixe))
                    # modified lines:
                    text = '%s\n' % date.day + '\n'.join([self.calevents[ev]['text'] for ev in ev_ids])
                    label.configure(text=text)

    def _show_event(self, date):
        """Display events on date if visible."""
        w, d = self._get_day_coords(date)
        if w is not None:
            label = self._calendar[w][d]
            if not label.cget('text'):
                # this is an other month's day and showothermonth is False
                return
            ev_ids = self._calevent_dates[date]
            i = len(ev_ids) - 1
            while i >= 0 and not self.calevents[ev_ids[i]]['tags']:
                i -= 1
            if i >= 0:
                tag = self.calevents[ev_ids[i]]['tags'][-1]
                label.configure(style='tag_%s.%s.TLabel' % (tag, self._style_prefixe))
            # modified lines:
            text = '%s\n' % date.day + '\n'.join([self.calevents[ev]['text'] for ev in ev_ids])
            label.configure(text=text)


class Html:
    def __init__(self, save_path: str):
        self.save_path = save_path

    def open_html(self):
        f = open(f'{self.save_path}', 'w')
        return f


if __name__ == '__main__':
    import tkinter as tk

    root = tk.Tk()
    root.geometry("800x500")
    agenda = Agenda(root, selectmode='none')
    topics = {
                'NN- AF, Backpropagation,': [],
                'soft/argmax, AdaBoost': [],
                'Ensemble models': [], # Gradient Boosting, XG Boost(Xtreme Gradient Boosting)
                'NN-multilayer perceptron': [],
                'SVM': [],

                'NN-autoencoder': [],
                'P-AUC,Rock': [], # Performance
                'P-micro/macro avg': [],
                'PR-Frequentist Vs Bayesian': [], # Pattern recognition
                'PR-LDA, hidden-markov': [],

                'PR-Kalman,Monte Carlo': [],
                '': [],
                'TS-VAR': [], # Time Series (TS)
                'TS-Holt-Winters': [],
                'TS-ARCH': [],

                'TS-Arima': [],
                'TS-Arima.imp': ['https://www.kaggle.com/zikazika/using-rnn-and-arima-to-predict-bitcoin-price'],
                'TS-Arima.imp': [],
                'AL-XGBoost': [], # algorithms (AL)
                'AL-CatBoost': [],

                'SM-over/under sampling': [], # sampling (SM)
                'SM-stratified sampling': [],
                'SM-augmentation.imp': [],
                'AL-alternating least squares': [],
                'AL-factorization algorithm': [],

                'centroid clustering': [],
                'anomaly detection': [],
                'KL Divergence': ['https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-understanding-kl-divergence-2b382ca2b2a8'],
                'negative sampling': [],
                '': [],
              }

    date = agenda.datetime.strptime('2021-10-21  1:33PM', '%Y-%m-%d %I:%M%p')
    # date = agenda.datetime.today() + agenda.timedelta(days=2)
    days = 0
    today_date = agenda.datetime.today()

    for topic in topics.keys():
        days += 1
        if today_date > (date + agenda.timedelta(days=days)):
            color = 'message'
        else:
            color = 'reminder'
        if ((date + agenda.timedelta(days=days)).weekday() == 3) or ((date + agenda.timedelta(days=days)).weekday() == 1):
            days += 1
            agenda.calevent_create(date + agenda.timedelta(days=days), topic, color)
        else:
            agenda.calevent_create(date + agenda.timedelta(days=days), topic, color)

    agenda.tag_config('reminder', background='red', foreground='white')

    agenda.pack(fill="both", expand=True)
    root.mainloop()
