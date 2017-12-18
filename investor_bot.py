from __future__ import print_function, division
import torch
import torch.autograd as autograd
import pandas
import sqlite3
import numpy as np
from tqdm import tqdm
import visdom
import random
vis = visdom.Visdom()

np.set_printoptions(suppress=True)

from InvestorNetwork import InvestorNetwork


def apply_exchange_fee(trade):
    bittrex_fee = 0.0025
    bittrex_fee = 0
    fee = bittrex_fee * trade
    return fee

def apply_network_fee(trade):
    return trade * 0.001

def p_normalize_columns(df):
    for col in df.columns:
        _col = df.loc[:, col].values
        if str(_col.dtype).startswith("float"):
            mean = _col.mean()
            std = _col.std()
            df.loc[:, col] -=mean
            df.loc[:, col] /= std
    return df

def normalize_columns(arr):
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return (arr - mean) / std


ACTIONS = [0, 1, 2]
DO_NOTHING, INVEST, WITHDRAW = 0, 1, 2
# State = [Current amount in fiat, Current amount in coin, Value of the ]
# Reward = Change in (Value(Fiat) + Value(Coin)) since last step

con = sqlite3.connect("coins_minute.db")
df = None

model = InvestorNetwork()

START_FUNDS = 1000

current_funds = START_FUNDS
current_coin = 0
time_range = 60
gamma = 0.99

def get_reward(pf, pc, p_cp, cf, cc, c_cp):
    p_fin = pf + (pc * p_cp)
    c_fin = cf + (cc * c_cp)
    return c_fin - p_fin

def get_total_value(coin_price):
    return current_funds.data[0] + (coin_price * current_coin.data[0])

def normalize_dollars(d):
    return d / START_FUNDS

def normalize_coin(c, coin_price):
    return c / coin_price

def invest(perc, coin_price):
    global current_funds, current_coin
    value = current_funds * perc
    current_coin = current_coin + (value / coin_price)
    current_funds = current_funds - value - apply_exchange_fee(value)
    #print("Buying %.3f coins (%.3f%% @ %.2f) I now have %.3f coins." % (value.data[0] / coin_price, perc.data[0], coin_price, current_coin.data[0]))

def withdraw(perc, coin_price):
    global current_funds, current_coin
    value = current_coin * perc
    current_coin = current_coin - value
    current_funds = current_funds + (value * coin_price) - apply_exchange_fee(value * coin_price)
    #print("Selling %.3f coins (%.3f%%) I now have %.2f$." % (value.data[0], perc.data[0], current_funds.data[0]))

def get_timeseries(df, ts):
    _timeseries = df.iloc[ts:ts+60, 2:].values.copy()
    timeseries = normalize_columns(_timeseries)
    timeseries = np.expand_dims(timeseries.T, axis=0)
    return timeseries, _timeseries

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def finish_episode():
    global model, gamma
    R = 0
    rewards = []
    for r in model.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for action, r in zip(model.saved_actions, rewards):
        action.reinforce(r)
    optimizer.zero_grad()
    autograd.backward(model.saved_actions, [None for _ in model.saved_actions], retain_graph=True)
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]
    return rewards.numpy()

def run_episode(start, end):
    global model, df
    reward_bias = (df.values[end, -3] - df.values[start, -3]) / (end - start) * 200
    for t, timestep in enumerate(tqdm(range(start, end))):
        timeseries, _timeseries = get_timeseries(df, timestep)
        x = autograd.Variable(torch.Tensor(timeseries))

        p_coin_price = _timeseries[-1, 3]
        action = model.forward(
            x,
            normalize_dollars(current_funds),
            normalize_coin(current_coin, p_coin_price)
        )

        to_invest_perc = action[0][0]
        to_withdraw_perc = action[0][1]

        pf, pc = (current_funds, current_coin)
        decisions = {
            True: "invest",
            False: "withdraw"
        }
        dec = to_invest_perc.data[0] > to_withdraw_perc.data[0]
        flip = False
        if random.random() < 0.05:
            dec = not dec
            flip = True
        decision = decisions[dec]

        model.saved_actions.append(action.multinomial())

        if decision == "invest":
            invest(to_invest_perc, p_coin_price)
        else:
            withdraw(to_withdraw_perc, p_coin_price)

        cf, cc = current_funds, current_coin
        c_coin_price = get_timeseries(df, timestep + 1)[1][-1, 3]

        reward = get_reward(pf, pc, p_coin_price, cf, cc, c_coin_price)
        tv = get_total_value(c_coin_price)
        done = tv < 200

        yield t, reward, action, done, cc, cf, tv, c_coin_price

def main():
    global model, current_coin, current_funds, df
    running_reward = 0
    coins = ["ETH"]
    episode_length = 1000
    bar = tqdm(range(100))
    for epoch in bar:
        for coin in coins:
            df = pandas.read_sql("SELECT * FROM coins_minute WHERE coin = '%s'" % coin, con)
            for i_episode in range(11):
                current_funds = START_FUNDS
                current_coin = 0

                ccs, cfs, tvs, cps = [], [], [], []
                start = i_episode * episode_length
                end = (i_episode + 1) * episode_length
                for step, reward, action, done, cc, cf, tv, cp in run_episode(start, end):
                    model.rewards.append(reward.data[0])
                    cfs.append(cf.data[0])
                    ccs.append(cc.data[0])
                    cps.append(cp)
                    #print(cp, cps[0], cp / cps[0], (cp / cps[0]) * 1000)
                    tvs.append([tv, 1000 * (cp / cps[0])])

                    if done:
                        break
                running_reward = running_reward * 0.99 + ((sum(model.rewards) / len(model.rewards)) * 0.01)
                n_rewards = model.rewards[:]
                q_rewards = finish_episode()[:len(n_rewards)]
                both_rewards = np.array([n_rewards, q_rewards]).T

                if i_episode == 5:
                    vis.line(both_rewards, np.arange(len(both_rewards)), opts={"title": "reward (%i)" % (epoch + 1)})
                    vis.line(np.array(ccs), np.arange(len(ccs)), opts={"title": "%s (%i)" % (coin, epoch + 1)})
                    vis.line(np.array(cfs), np.arange(len(cfs)), opts={"title": "Funds (%i)" % (epoch + 1)})
                    vis.line(np.array(tvs), np.arange(len(tvs)), opts={"title": "Total value (%i,%i)" % (epoch + 1, i_episode + 1)})


                #print("Episode %i, last length %i steps." % (i_episode, step))
            bar.write("Running reward: %.3f" % running_reward)

if __name__ == "__main__":
    main()
