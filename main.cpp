#include <iostream>
#include <vector>
#include <fstream>
#include "Bandit.h"

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
bandit_simulation(int num_bandits, int time_step, std::vector<std::vector<Bandit *>> &bandits);

void epsilon_greedy(int num_bandits, int time_step, int k_arms, bool bernoulli);

void optimistic_initial_values(int num_bandits, int time_step, int k_arms, bool bernoulli);

void gradient_bandit(int num_bandits, int time_step, int k_arms, bool bernoulli);

void ucb(int num_bandits, int time_step, int k_arms, bool bernoulli);


void send_to_file(const std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> &res,
                  const std::string &params);

void clear_bandits(const std::vector<std::vector<Bandit *>> &bandits);

int main() {
    bool bernoulli[] = {true, false};
    for (auto &b : bernoulli) {
        epsilon_greedy(10000, 1000, 20, b);
        optimistic_initial_values(10000, 1000, 20, b);
        ucb(10000, 1000, 20, b);
        gradient_bandit(10000, 1000, 20, b);
    }
    return 0;
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
bandit_simulation(int num_bandits, int time_step, std::vector<std::vector<Bandit *>> &bandits) {
    std::vector<std::vector<double>> best_action_counts(bandits.size(), std::vector<double>(time_step, 0.0));
    std::vector<std::vector<double>> average_rewards(bandits.size(), std::vector<double>(time_step, 0.0));
    for (size_t b = 0; b < bandits.size(); b++) {
        for (int i = 0; i < num_bandits; i++) {
            for (int t = 0; t < time_step; t++) {
                auto action_index = bandits[b][i]->get_action();
                auto reward = bandits[b][i]->take_action(action_index);
                average_rewards[b][t] += reward;
                if (action_index == bandits[b][i]->best_action) best_action_counts[b][t]++;
            }
        }
        for (auto &x : best_action_counts[b]) x = x / num_bandits;
        for (auto &x : average_rewards[b]) x = x / num_bandits;

    }
    return make_pair(best_action_counts, average_rewards);
}

void epsilon_greedy(int num_bandits, int time_step, int k_arms, bool bernoulli) {
    std::vector<double> epsilons = {0, 0.1, 0.05, 0.01};
    std::vector<std::vector<Bandit *>> all_bandit_experiments;
    for (double epsilon : epsilons) {
        std::vector<Bandit *> bandits;
        for (int j = 0; j < num_bandits; j++) {
            bandits.push_back(new Bandit(k_arms, epsilon, 0., bernoulli));
            bandits.back()->epsilon = epsilon;
            bandits.back()->sample_averages = true;
        }
        all_bandit_experiments.push_back(bandits);
    }
    auto res = bandit_simulation(num_bandits, time_step, all_bandit_experiments);
    std::string filename = "epsilon_" + std::to_string(k_arms) + (bernoulli ? "_bernoulli" : "_normal");
    send_to_file(res, filename);
    clear_bandits(all_bandit_experiments);
}


void optimistic_initial_values(int num_bandits, int time_step, int k_arms, bool bernoulli) {
    using namespace std;
    vector<vector<Bandit *>> all_bandit_experiments;
    vector<Bandit *> bandits;
    int optimistic[] = {0, 1, 2, 5};
    for (auto &o:optimistic) {
        for (int j = 0; j < num_bandits; j++) {
            bandits.push_back(new Bandit(k_arms, 0., o, bernoulli, 0.1));
        }
        all_bandit_experiments.push_back(bandits);
        bandits.clear();
    }
    auto res = bandit_simulation(num_bandits, time_step, all_bandit_experiments);
    std::string filename = "optimistic_" + std::to_string(k_arms) + (bernoulli ? "_bernoulli" : "_normal");
    send_to_file(res, filename);
    clear_bandits(all_bandit_experiments);
}

void ucb(int num_bandits, int time_step, int k_arms, bool bernoulli) {
    std::vector<std::vector<Bandit *>> all_bandit_experiments;
    std::vector<Bandit *> bandits;
    double ucb_param[] = {0, 0.5, 1, 1.5, 2};
    for (auto &ucb : ucb_param) {
        for (int j = 0; j < num_bandits; j++) {
            bandits.push_back(new Bandit(k_arms, 0., 0., bernoulli,
                                         0.1, false, &ucb));
        }
        all_bandit_experiments.push_back(bandits);
        bandits.clear();
    }
    auto res = bandit_simulation(num_bandits, time_step, all_bandit_experiments);
    std::string filename = "ucb_" + std::to_string(k_arms) + (bernoulli ? "_bernoulli" : "_normal");
    send_to_file(res, filename);
    clear_bandits(all_bandit_experiments);
}

void gradient_bandit(int num_bandits, int time_step, int k_arms, bool bernoulli) {
    std::vector<std::vector<Bandit *>> all_bandit_experiments;
    std::vector<Bandit *> bandits;
    bool baselines[] = {true, false};
    double step_sizes[] = {0.05, 0.1, 0.4};
    for (auto &baseline : baselines) {
        for (auto &step_size : step_sizes) {
            for (int j = 0; j < num_bandits; j++) {
                bandits.push_back(new Bandit(k_arms, 0., 0., bernoulli,
                                             step_size, false, nullptr,
                                             true, baseline));
            }
            all_bandit_experiments.push_back(bandits);
            bandits.clear();
        }
    }
    auto res = bandit_simulation(num_bandits, time_step, all_bandit_experiments);
    std::string filename = "gradient_" + std::to_string(k_arms) + (bernoulli ? "_bernoulli" : "_normal");
    send_to_file(res, filename);
    clear_bandits(all_bandit_experiments);
}

void send_to_file(const std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> &res,
                  const std::string &params) {
    std::ofstream output;
    std::string name = "bandit_" + params;
    output.open(name);
    for (auto &v : res.first) {
        for (auto &u: v) {
            output << u << " ";
        }
        output << std::endl;
    }
    output << "\n\n";
    for (auto &v : res.second) {
        for (auto &u: v) {
            output << u << " ";
        }
        output << std::endl;
    }
    output << "\n" << std::endl;
    output.close();
}

void clear_bandits(const std::vector<std::vector<Bandit *>> &bandits) {
    for (auto &v : bandits)
        for (auto p : v)
            delete p;
}

