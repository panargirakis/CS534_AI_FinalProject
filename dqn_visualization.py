import plotly.graph_objects as go
import pandas as pd
import os
import argparse

#  define write files argument

parser = argparse.ArgumentParser(description='Visualize q learning data')
parser.add_argument('--write-files', choices=['y', 'n'], default='n',
                    help='a boolean, whether to write to png')
args = parser.parse_args()
WRITE_FILES = args.write_files == "y"

#  create file for visualizations

viz_folder = "dqn_visualizations"
if WRITE_FILES and not os.path.exists(viz_folder):
    os.mkdir(viz_folder)

# setup which models to visualize

models = [("standard_dqn", "dqn_atari_BreakoutDeterministic-v4_log.json"),
          ("dueling_dqn", "dueldqn_atari_BreakoutDeterministic-v4_log.json"),
          ("double_dqn", "ddqn_atari_BreakoutDeterministic-v4_log.json"),
          ("double_dueling_dqn", "double_duel_dqn_BreakoutDeterministic-v4_log.json")]
data_folder = "checkpoints"

# create graphs for each model

for model_name, model_path in models:

    # read data
    df = pd.read_json("{}/{}".format(data_folder, model_path))

    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["episode"], y=df["episode_reward"],
                             mode='lines',
                             name='Episide Reward'))
    fig.add_trace(go.Scatter(x=df["episode"], y=df["episode_reward"].rolling(100).mean(),
                             mode='lines',
                             name='Ep. Reward (Smoothed)'))
    fig.add_trace(go.Scatter(x=df["episode"], y=df["duration"].rolling(100).mean(),
                             mode='lines',
                             name='Ep. Duration (Smoothed, in s)'))

    fig.update_layout(
        title="Reward and Duration vs. Episode - {}".format(model_name),
        xaxis=dict(
            title="Episode",
            dtick=500  # make x ticks more dense
        ),
        yaxis=dict(
            dtick=5  # make x ticks more dense
        )
    )

    if WRITE_FILES:
        fig.write_image("{}/{}_rwd_dur.png".format(viz_folder, model_name))
    else:
        fig.show()
        # pass

    fig.data = []
    fig.update_layout(title="Loss vs. Episode - {}".format(model_name), yaxis=dict(title="Loss", dtick=None))

    fig.add_trace(go.Scatter(x=df["episode"], y=df["loss"],
                             mode='lines',
                             name='Loss'))
    fig.add_trace(go.Scatter(x=df["episode"], y=df["loss"].rolling(100).mean(),
                             mode='lines',
                             name='Loss (Smoothed)'))

    if WRITE_FILES:
        fig.write_image("{}/{}_loss.png".format(viz_folder, model_name))
    else:
        fig.show()
