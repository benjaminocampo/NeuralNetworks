import os
import itertools as it
import numpy as np
import pandas as pd
import re
from time import time
from os import makedirs
from os.path import exists

printf = lambda s: print(s, flush=True)

cmd_error_count = 0

DIR_STDOUTS = 'runs/stdouts'
DIR_CHARTS = 'runs/charts'


def cmd(c):
    global cmd_error_count
    red_color = "\033[91m"
    end_color = "\033[0m"
    printf(f"\n>>> [COMMAND] {c} @ {os.getcwd()}")
    if os.system(c):  # Command returned != 0
        printf(f"{red_color}>>> [ERROR] there was an error in command:{end_color}")
        printf(f"{red_color}>>> [ERROR] {c} @ {os.getcwd()}{end_color}")
        cmd_error_count += 1
        exit()


def read_netstats(filename):
    params_tag = '[PARAMS]'
    epoch_tag = '[EPOCH]'
    stats = {}
    stats['train_avgloss'] = []
    stats['test_avgloss'] = []
    stats['epoch_time'] = []
    with open(filename, 'r') as logfile:
        for line in logfile:
            if line.startswith(epoch_tag):
                epoch_vars = re.findall(r'([\w]+)=', line)
                epoch_values = re.findall(r'=([\w.]+)', line)
                for var, value in zip(epoch_vars, epoch_values):
                    if var != 'epoch':
                        stats[var].append(value)
            if line.startswith(params_tag):
                param_vars = re.findall(r'([\w]+)=', line)
                params_values = re.findall(r'=([\w.]+)', line)
                for var, value in zip(param_vars, params_values):
                    stats[var] = value
    # Last train and test loss are important
    stats['train_avgloss'] = float(stats['train_avgloss'][-1])
    stats['test_avgloss'] = float(stats['test_avgloss'][-1])
    stats['nof_epochs'] = int(stats['nof_epochs'])
    # But we need an average of the time needed by each epoch
    stats['epoch_time'] = np.mean([float(time) for time in stats['epoch_time']])
    return pd.Series(stats)


class Runner:
    is_run_folder_initialized = False

    def __init__(
        self,
        nof_epochs,
        batch_size,
        optimizer,
        learning_rate,
        momentum,
    ):
        self.nof_epochs = nof_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum

        if not Runner.is_run_folder_initialized:
            Runner.setup_run_folder()
            Runner.is_run_folder_initialized = True

    @staticmethod
    def setup_run_folder():
        makedirs(DIR_STDOUTS, exist_ok=True)
        makedirs(DIR_CHARTS, exist_ok=True)

    @staticmethod
    def sanitize_filename(value):
        if isinstance(value, dict):
            return "__".join(
                f"{Runner.sanitize_filename(k)}--{Runner.sanitize_filename(v)}"
                for k, v in value.items()
                if v
            )
        elif isinstance(value, str):
            isvalid = lambda l: l.isalnum() or l in ["-", "_"]
            return "".join(letter if isvalid(letter) else "-" for letter in value)
        return value

    def __str__(self):
        return Runner.sanitize_filename(vars(self))

    @property
    def run_name(self):
        return str(self)

    @property
    def run_cmd(self):
        params = ''
        for k, v in vars(self).items():
            params += f'{k}={v}\t'
        run_cmd = f'echo "[PARAMS] {params}" > runs/stdouts/{self.run_name}.output && '\
                  f'python3 autoencoder.py '\
                  f'-epochs {self.nof_epochs} '\
                  f'-bs {self.batch_size} '\
                  f'-opt {self.optimizer} '\
                  f'-lr {self.learning_rate} '\
                  f'-m {self.momentum} '\
                  f'>> runs/stdouts/{self.run_name}.output'
        return run_cmd

    def run(self):
        cmd(self.run_cmd)


def main():
    nsof_epochs = [8, 16]
    batch_sizes = [128, 256, 1000]
    optimizers = ['SGD', 'ADAM']
    learning_rates = [1e-1, 1e-2, 1e-3]
    momentums = [.2, .4, .6, .8]

    runs = it.product(
        nsof_epochs,
        batch_sizes,
        optimizers,
        learning_rates,
        momentums,
    )

    itime = time()
    printf('>>> [START]')
    all_runs_dataframes = []
    for (run_counter, (nof_epochs, batch_size, optimizer, learning_rate, momentum)) in enumerate(runs):
        itime_run = time()
        runner = Runner(
            nof_epochs=nof_epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            learning_rate=learning_rate,
            momentum=momentum
        )
        runner.run()
        filename = f'{runner.run_name}.output'
        stdout_file = f'{DIR_STDOUTS}/{filename}'

        if exists(stdout_file):
            run_metrics = read_netstats(stdout_file)
            all_runs_dataframes.append(run_metrics)
        printf(f'[RUN] [{run_counter}] Run finished in {time() - itime_run:3f}s')

    html = pd.DataFrame()
    html = html.append(all_runs_dataframes).to_html()
    text_file = open(f"run_table.html", "w")
    text_file.write(html)
    text_file.close()

    printf(f">>> [END] Done in {time() - itime:3f} seconds with {cmd_error_count} errors.")


if __name__ == "__main__":
    main()
