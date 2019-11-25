#include <iostream>
#include <vector>
#include "Bandit.h"
#include "vector"

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
bandit_simulation(int num_bandits, int time_step, std::vector<std::vector<Bandit>> &bandits) {
    std::vector<std::vector<double>> best_action_counts(bandits.size(), std::vector<double>(time_step, 0.0));
    std::vector<std::vector<double>> average_rewards(bandits.size(), std::vector<double>(time_step, 0.0));
    for (size_t k = 0; k < bandits.size(); k++) {
        for (int i = 0; i < num_bandits; i++) {
            for (int t = 0; t < time_step; t++) {
                auto action_index = bandits[k][i].action();
                auto reward = bandits[k][i].step(action_index);
                average_rewards[k][t] += reward;
                if (action_index == bandits[k][i].best_action) best_action_counts[k][t] += 1;
            }
        }
        for (auto &x : best_action_counts[k]) x = x / num_bandits;
        for (auto &x : average_rewards[k]) x = x / num_bandits;
    }
    return make_pair(best_action_counts, average_rewards);
}

void epsilon_greedy(int num_bandits, int time_step) {
    std::vector<double> epsilons = {0, 0.1, 0.01};
    std::vector<std::vector<Bandit>> bandits;
    for (double epsilon : epsilons) {
        std::vector<Bandit> vec_bandits;
        for (int j = 0; j < num_bandits; j++) {
            vec_bandits.emplace_back(10);
            vec_bandits.back().epsilon = epsilon;
            vec_bandits.back().sample_averages = true;
        }
        bandits.push_back(vec_bandits);
    }
    auto res = bandit_simulation(num_bandits, time_step, bandits);
}

void optimistic_initial_values(int num_bandits, int time_step) {
    std::vector<std::vector<Bandit>> bandits;
    std::vector<Bandit> vec_bandits(num_bandits);
    for (int j = 0; j < num_bandits; j++) {
        vec_bandits.emplace_back(10, 0., 5, nullptr, 0.1);
    }
    bandits.push_back(vec_bandits);
    vec_bandits.clear();
    for (int j = 0; j < num_bandits; j++) {
        vec_bandits.emplace_back(10, 0.1, 0, nullptr, 0.1);
    }
    bandits.push_back(vec_bandits);
    auto res = bandit_simulation(num_bandits, time_step, bandits);
}

void ucb(int num_bandits, int time_step) {
    std::vector<std::vector<Bandit>> bandits;
    std::vector<Bandit> vec_bandits(num_bandits);
    double ucb_param = 2;
    for (int j = 0; j < num_bandits; j++) {
        vec_bandits.emplace_back(10, 0., 0., nullptr,
                                 0.1, false, &ucb_param);
    }
    bandits.push_back(vec_bandits);
    vec_bandits.clear();
    for (int j = 0; j < num_bandits; j++) {
        vec_bandits.emplace_back(10, 0.1, 0., nullptr,
                                 0.1);
    }
    bandits.push_back(vec_bandits);
    auto res = bandit_simulation(num_bandits, time_step, bandits);

}

void gradient_bandit(int num_bandits, int time_step) {
    std::vector<std::vector<Bandit>> bandits;
    std::vector<Bandit> vec_bandits(num_bandits);
    for (int j = 0; j < num_bandits; j++) {
        vec_bandits.emplace_back(10, 0., 0., nullptr,
                                 0.1, false, nullptr,
                                 true, true, 4);
    }
    bandits.push_back(vec_bandits);
    vec_bandits.clear();
    for (int j = 0; j < num_bandits; j++) {
        vec_bandits.emplace_back(10, 0., 0., nullptr,
                                 0.1, false, nullptr,
                                 true, false, 4);
    }
    bandits.push_back(vec_bandits);
    vec_bandits.clear();
    for (int j = 0; j < num_bandits; j++) {
        vec_bandits.emplace_back(10, 0., 0., nullptr,
                                 0.4, false, nullptr,
                                 true, true, 4);
    }
    bandits.push_back(vec_bandits);
    vec_bandits.clear();
    for (int j = 0; j < num_bandits; j++) {
        vec_bandits.emplace_back(10, 0., 0., nullptr,
                                 0.4, false, nullptr,
                                 true, false, 4);
    }
    bandits.push_back(vec_bandits);
    auto res = bandit_simulation(num_bandits, time_step, bandits);
}

int main() {
    return 0;
}